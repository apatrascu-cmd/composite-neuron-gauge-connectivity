#!/usr/bin/env python3
"""Capacity- and initialisation-matched interpolation of the phase-routing read-out.

lambda=1 is the invariant modulus used by PhaseRouting.  lambda=0 is an
absolute real projection and lambda=0.5 is the fixed interior point.  A fixed
factor 2/(1+lambda) equalises the expected squared output for rotationally
balanced complex inputs, so lambda does not also change the initial activation
scale.  All
three values are rerun by this script so that a given seed has identical
initial weights, data order and optimiser state across lambda.  The
architecture, parameter count, data split, optimiser, regulariser and training
length match the archived adaptive_phase configuration.  This experiment asks
whether clean accuracy follows the invariant read-out; it is not an
image-adversarial-robustness experiment.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_cifar10_composite import (
    CompositeRegulariser,
    accuracy,
    cluster_compactness,
    load_cifar10,
)


HERE = os.path.dirname(os.path.abspath(__file__))


def state_sha256(*modules):
    """Stable fingerprint of initial tensors; used to verify paired initialisation."""
    h = hashlib.sha256()
    for prefix, module in enumerate(modules):
        for name, value in sorted(module.state_dict().items()):
            a = value.detach().cpu().contiguous().numpy()
            h.update(('{}:{}:{}:{}'.format(prefix, name, a.dtype, a.shape)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


class EllipticPhaseRouting(nn.Module):
    """Same complex map as PhaseRouting with a fixed anisotropic read-out."""
    def __init__(self, n_in, n_out, readout_lambda):
        super().__init__()
        self.m = n_in // 2
        self.readout_lambda = float(readout_lambda)
        self.w = nn.Parameter(torch.randn(n_out, self.m) / np.sqrt(self.m))
        self.theta = nn.Parameter(2 * np.pi * torch.rand(n_out, self.m))

    def complex_output(self, x):
        cr, ci = x[:, :self.m], x[:, self.m:2 * self.m]
        Wr, Wi = self.w * torch.cos(self.theta), self.w * torch.sin(self.theta)
        yr = cr @ Wr.t() - ci @ Wi.t()
        yi = cr @ Wi.t() + ci @ Wr.t()
        return yr, yi

    def forward(self, x):
        yr, yi = self.complex_output(x)
        moment_scale = 2.0 / (1.0 + self.readout_lambda)
        return torch.sqrt(moment_scale *
                          (yr * yr + self.readout_lambda * yi * yi) + 1e-8)


class AblationNet(nn.Module):
    def __init__(self, readout_lambda):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.routing = EllipticPhaseRouting(64 * 6 * 6, 64, readout_lambda)
        self.dense = nn.Linear(64, 64)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(64, 10)

    def logits_from_features(self, h):
        return self.out(self.drop(F.relu(self.dense(self.routing(h)))))

    def forward(self, x):
        return self.logits_from_features(self.features(x))


def rotate_paired_features(h, beta):
    m = h.shape[1] // 2
    cr, ci = h[:, :m], h[:, m:2 * m]
    cb, sb = np.cos(beta), np.sin(beta)
    return torch.cat((cb * cr - sb * ci, sb * cr + cb * ci), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--readout_lambda', type=float, required=True, choices=[0.0, 0.5, 1.0])
    ap.add_argument('--gamma', type=float, default=1e-2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--phase_probe_n', type=int, default=512)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    xtr, ytr, xte, yte = load_cifar10()
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(xtr))
    val_idx, tr_idx = perm[:5000], perm[5000:]
    if args.quick:
        tr_idx, val_idx, xte, yte = tr_idx[:4000], val_idx[:1000], xte[:1000], yte[:1000]
        args.epochs = min(args.epochs, 1)
    Xtr, Ytr = torch.tensor(xtr[tr_idx]), torch.tensor(ytr[tr_idx])
    Xva, Yva = torch.tensor(xtr[val_idx]), torch.tensor(ytr[val_idx])
    Xte, Yte = torch.tensor(xte), torch.tensor(yte)

    model = AblationNet(args.readout_lambda)
    clusters = [list(range(10 * j, 10 * (j + 1))) for j in range(4)]
    elementary = list(range(40, 64))
    reg = CompositeRegulariser(tuple(model.dense.weight.shape), clusters, elementary, adaptive=True)
    reg.init_from(model.dense.weight.detach())
    initial_state_hash = state_sha256(model, reg)
    split_hash = hashlib.sha256(np.asarray(perm, dtype=np.int64).tobytes()).hexdigest()
    # Record the activation scale before any optimisation.  This is a direct
    # guard against mistaking a lambda-dependent signal amplitude for an
    # invariance effect.  Evaluation has no stochastic layer and consumes no
    # random numbers, so it leaves the paired training trajectories intact.
    model.eval()
    with torch.no_grad():
        h_initial = model.features(Xva[:min(512, len(Xva))])
        yr_initial, yi_initial = model.routing.complex_output(h_initial)
        routing_initial = model.routing(h_initial)
        initial_yr_second_moment = float(torch.mean(yr_initial * yr_initial))
        initial_yi_second_moment = float(torch.mean(yi_initial * yi_initial))
        initial_routing_rms = float(torch.sqrt(torch.mean(routing_initial * routing_initial)))
    params = list(model.parameters()) + [p for p in reg.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, 1e-3)

    lam_tag = str(args.readout_lambda).replace('.', 'p')
    tag = 'cnn_phase_lambda{}_g{:g}_s{}'.format(lam_tag, args.gamma, args.seed)
    if args.quick:
        tag += '_quick'
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    logf = open(os.path.join(HERE, 'results', tag + '.log'), 'w')

    def log(message):
        line = '[{}] {} {}'.format(time.strftime('%H:%M:%S'), tag, message)
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    bs = 64
    n_batches = (len(Xtr) + bs - 1) // bs
    history = []
    start = time.time()
    log('lambda={} seed={} epochs={} train={} val={} params={}'.format(
        args.readout_lambda, args.seed, args.epochs, len(Xtr), len(Xva),
        sum(p.numel() for p in model.parameters())))
    for epoch in range(args.epochs):
        model.train()
        order = torch.tensor(rng.permutation(len(Xtr)))
        total_loss = total_penalty = 0.0
        correct = 0
        for b in range(n_batches):
            idx = order[b * bs:(b + 1) * bs]
            xb, yb = Xtr[idx], Ytr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            penalty = reg.penalty(model.dense.weight)
            objective = loss + args.gamma * penalty
            opt.zero_grad(); objective.backward(); opt.step()
            total_loss += loss.item() * len(idx)
            total_penalty += penalty.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        validation = accuracy(model, Xva, Yva)
        elapsed = time.time() - start
        rec = {'epoch': epoch + 1, 'train_loss': total_loss / len(Xtr),
               'penalty': total_penalty / len(Xtr),
               'train_acc': correct / len(Xtr), 'val_acc': validation,
               'elapsed_s': elapsed}
        history.append(rec)
        log('epoch {}/{} loss {:.4f} train {:.4f} val {:.4f} ETA {:.1f} min'.format(
            epoch + 1, args.epochs, rec['train_loss'], rec['train_acc'], validation,
            elapsed / (epoch + 1) * (args.epochs - epoch - 1) / 60.0))

    model.eval()
    clean = accuracy(model, Xte, Yte)
    nprobe = min(args.phase_probe_n, len(Xte))
    with torch.no_grad():
        h = model.features(Xte[:nprobe])
        r0 = model.routing(h)
        l0 = model.logits_from_features(h)
        hr = rotate_paired_features(h, 0.731)
        r1 = model.routing(hr)
        l1 = model.logits_from_features(hr)
        route_rms = torch.sqrt(torch.mean((r1 - r0) ** 2))
        route_scale = torch.sqrt(torch.mean(r0 ** 2)) + 1e-12
        logit_rms = torch.sqrt(torch.mean((l1 - l0) ** 2))
        logit_scale = torch.sqrt(torch.mean(l0 ** 2)) + 1e-12
    result = {
        'readout_lambda': args.readout_lambda,
        'readout_second_moment_scale': 2.0 / (1.0 + args.readout_lambda),
        'gamma': args.gamma,
        'seed': args.seed,
        'epochs': args.epochs,
        'quick': args.quick,
        'history': history,
        'clean_test_acc': clean,
        'val_acc_final': history[-1]['val_acc'],
        'cluster_compactness': cluster_compactness(model.dense.weight.detach(), clusters),
        'regulariser_stats': reg.stats(model.dense.weight.detach()),
        'phase_probe_n': nprobe,
        'phase_probe_beta': 0.731,
        'routing_common_phase_max_deviation': float(torch.max(torch.abs(r1 - r0))),
        'routing_common_phase_relative_rms_deviation': float(route_rms / route_scale),
        'logit_common_phase_max_deviation': float(torch.max(torch.abs(l1 - l0))),
        'logit_common_phase_relative_rms_deviation': float(logit_rms / logit_scale),
        'prediction_agreement_under_common_phase': float((l1.argmax(1) == l0.argmax(1)).float().mean()),
        'n_params': sum(p.numel() for p in model.parameters()),
        'initial_state_sha256': initial_state_hash,
        'split_sha256': split_hash,
        'initial_yr_second_moment': initial_yr_second_moment,
        'initial_yi_second_moment': initial_yi_second_moment,
        'initial_routing_rms': initial_routing_rms,
        'wall_time_s': time.time() - start,
        'code_sha256': hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
        'base_code_sha256': hashlib.sha256(open(os.path.join(HERE, 'cnn_cifar10_composite.py'), 'rb').read()).hexdigest(),
    }
    with open(os.path.join(HERE, 'results', tag + '.json'), 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    log('clean {:.4f} route_phase_defect {:.6g} logit_phase_defect {:.6g}; DONE {:.1f} min'.format(
        clean, result['routing_common_phase_max_deviation'],
        result['logit_common_phase_max_deviation'], result['wall_time_s'] / 60.0))


if __name__ == '__main__':
    main()
