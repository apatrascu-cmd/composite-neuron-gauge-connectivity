#!/usr/bin/env python3
"""
CIFAR-10 experiment: composite (soft-shared) regularisation of a small CNN, with and without a
routing layer, evaluated on clean, FGSM and PGD test accuracy.  PyTorch >= 1.9, CPU is enough.

Configurations (identical backbone Conv32-Pool-Conv64-Pool-Flatten-[routing]-Dense64-Dropout-Dense10):
  baseline      : no regulariser
  semicomp      : semi-composite regulariser on the Dense64 layer: 4 composite clusters of 10 units with
                  learned centroids Lambda_j (the composite operators) and 24 elementary units whose effective
                  centroid is the uniform mixture of the four Lambda_j
  adaptive      : as semicomp, but each elementary unit learns its mixing weights alpha_uj = softmax(a_uj)
  adaptive_girl : adaptive + the cosine-parametrised real routing layer of the original code (W = cos(G))
  adaptive_phase: adaptive + phase-routing layer with modulus read-out y_k = | sum_j w_kj e^{i theta_kj} c_j |
                  (invariant under a common rephasing of its complex inputs; covariant under per-input rephasing)

Penalty for a unit u with incoming weight vector w_u:  gamma * sum_u || w_u - Lambda^eff_u ||^2,
Lambda^eff_u = Lambda_j for u in cluster j, and sum_j alpha_uj Lambda_j for elementary u.

Usage: python cnn_cifar10_composite.py --config adaptive --gamma 1e-2 --seed 0 [--epochs 10] [--quick]
Results: results/cnn_<config>_g<gamma>_s<seed>.json  (all numbers in the paper come from these files)
"""
import argparse, hashlib, json, os, sys, time, pickle, tarfile, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def load_cifar10():
    cands = [os.path.join(HERE, 'data', 'cifar-10-batches-py'), os.path.expanduser('~/.keras/datasets/cifar-10-batches-py')]
    d = next((c for c in cands if os.path.isdir(c)), None)
    if d is None:
        os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
        tgz = os.path.join(HERE, 'data', 'cifar-10-python.tar.gz')
        print('downloading CIFAR-10 (163 MB) ...', flush=True); urllib.request.urlretrieve(CIFAR_URL, tgz)
        tarfile.open(tgz).extractall(os.path.join(HERE, 'data')); d = cands[0]
    def batch(f):
        b = pickle.load(open(os.path.join(d, f), 'rb'), encoding='bytes')
        return b[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0, np.array(b[b'labels'])
    xs, ys = zip(*[batch(f'data_batch_{i}') for i in range(1, 6)])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    xte, yte = batch('test_batch')
    return xtr, ytr, xte, yte

# ----------------------------------------------------------------------------- layers
class CosRouting(nn.Module):
    """the original 'GaugeInvariantRouting' layer: a real linear map with weights cos(G)/sqrt(n_in/2), |w| <= 1.
    The original code initialised G ~ N(0,1).  Since E[cos G] = exp(-1/2) = 0.61 for G ~ N(0,1), all n_out outputs were then
    nearly the same multiple of the summed input, about sqrt(n_in) times too large, and the layer did not train (chance-level
    accuracy for 2 epochs, log kept as results/cnn_adaptive_girl_g0.01_s0_v1_unscaled_init.log).  We initialise G ~ U(0, 2 pi)
    (zero-mean weights) and scale by 1/sqrt(n_in/2) (unit-variance outputs); init='normal_unscaled' reproduces the original."""
    def __init__(self, n_in, n_out, init='uniform'):
        super().__init__(); self.init = init
        self.G = nn.Parameter(torch.randn(n_in, n_out) if init == 'normal_unscaled' else 2 * np.pi * torch.rand(n_in, n_out))
        self.scale = 1.0 if init == 'normal_unscaled' else 1.0 / np.sqrt(n_in / 2)
    def forward(self, x): return (x @ torch.cos(self.G)) * self.scale

class PhaseRouting(nn.Module):
    """complex routing with modulus read-out.  Inputs (real, dim 2m) are paired into m complex numbers
    c = x[:m] + i x[m:]; output_k = | sum_j w_kj exp(i theta_kj) c_j |.  A common rephasing of c leaves the
    output invariant; a per-input rephasing c_j -> e^{i a_j} c_j is absorbed by theta_kj -> theta_kj - a_j."""
    def __init__(self, n_in, n_out):
        super().__init__(); m = n_in // 2; self.m = m
        self.w = nn.Parameter(torch.randn(n_out, m) / np.sqrt(m)); self.theta = nn.Parameter(2 * np.pi * torch.rand(n_out, m))
    def forward(self, x):
        cr, ci = x[:, :self.m], x[:, self.m:2 * self.m]
        Wr, Wi = self.w * torch.cos(self.theta), self.w * torch.sin(self.theta)
        yr = cr @ Wr.t() - ci @ Wi.t(); yi = cr @ Wi.t() + ci @ Wr.t()
        return torch.sqrt(yr * yr + yi * yi + 1e-8)

class Net(nn.Module):
    def __init__(self, routing=None):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        n_flat = 64 * 6 * 6
        self.routing = {None: None, 'cos': CosRouting(n_flat, 64), 'phase': PhaseRouting(n_flat, 64)}[routing]
        self.dense = nn.Linear(64 if routing else n_flat, 64)      # the regularised layer (rows = units)
        self.drop = nn.Dropout(0.5); self.out = nn.Linear(64, 10)
    def forward(self, x):
        h = self.features(x)
        if self.routing is not None: h = self.routing(h)
        return self.out(self.drop(F.relu(self.dense(h))))

class CompositeRegulariser(nn.Module):
    """semi-composite / adaptive semi-composite penalty on the rows of a weight matrix."""
    def __init__(self, W_shape, clusters, elementary, adaptive):
        super().__init__()
        n_units, n_in = W_shape; self.clusters = clusters; self.elementary = elementary; self.adaptive = adaptive
        self.Lambda = nn.Parameter(torch.zeros(len(clusters), n_in))          # composite operators (centroids)
        self.a = nn.Parameter(torch.zeros(len(elementary), len(clusters)), requires_grad=adaptive)  # mixing logits
        self.register_buffer('Lambda0', torch.zeros(len(clusters), n_in))
    def init_from(self, W):
        with torch.no_grad():
            for j, c in enumerate(self.clusters): self.Lambda[j] = W[c].mean(0)
            self.Lambda0.copy_(self.Lambda)
    def alpha(self): return torch.softmax(self.a, dim=1)
    def penalty(self, W):
        pen = sum(((W[c] - self.Lambda[j]) ** 2).sum() for j, c in enumerate(self.clusters))
        Leff = self.alpha() @ self.Lambda                                     # (n_elem, n_in)
        pen = pen + ((W[self.elementary] - Leff) ** 2).sum()
        return pen
    def stats(self, W):
        with torch.no_grad():
            al = self.alpha(); ent = -(al * torch.log(al + 1e-12)).sum(1).mean() / np.log(al.shape[1])
            return {'Lambda_moved_norm': float((self.Lambda - self.Lambda0).norm(dim=1).mean()), 'Lambda_norm': float(self.Lambda.norm(dim=1).mean()),
                    'alpha_mean_rows': al.mean(0).tolist(), 'alpha_normalised_entropy': float(ent), 'alpha_max_mean': float(al.max(1).values.mean())}

def cluster_compactness(W, clusters):
    """1 - (within-cluster dispersion about the empirical centroids) / (total dispersion of the composite units)."""
    with torch.no_grad():
        comp = torch.cat([W[c] for c in clusters]); tot = ((comp - comp.mean(0)) ** 2).sum()
        within = sum(((W[c] - W[c].mean(0)) ** 2).sum() for c in clusters)
        return float(1 - within / (tot + 1e-12))

# ----------------------------------------------------------------------------- attacks
def fgsm(model, x, y, eps):
    x = x.clone().requires_grad_(True); loss = F.cross_entropy(model(x), y); g, = torch.autograd.grad(loss, x)
    return (x + eps * g.sign()).clamp(0, 1).detach()

def pgd(model, x, y, eps, steps=10, alpha=None, rng=None):
    alpha = alpha or eps / 4
    xa = (x + eps * (2 * torch.rand_like(x) - 1)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_(True); loss = F.cross_entropy(model(xa), y); g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * g.sign(); xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa.detach()

def accuracy(model, x, y, attack=None, bs=500, **kw):
    model.eval(); correct = 0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        if attack is not None: xb = attack(model, xb, yb, **kw)
        with torch.no_grad(): correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(x)

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='baseline', choices=['baseline', 'semicomp', 'adaptive', 'adaptive_girl', 'adaptive_phase'])
    ap.add_argument('--gamma', type=float, default=1e-2); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--eps', type=float, default=0.03); ap.add_argument('--threads', type=int, default=2); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n_adv', type=int, default=2000, help='number of test images for the FGSM/PGD evaluation (first n_adv of the test set)')
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    xtr, ytr, xte, yte = load_cifar10()
    rng = np.random.default_rng(a.seed); perm = rng.permutation(len(xtr))
    n_val = 5000; val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if a.quick: tr_idx, val_idx, xte, yte = tr_idx[:4000], val_idx[:1000], xte[:1000], yte[:1000]; a.epochs = min(a.epochs, 2)
    Xtr, Ytr = torch.tensor(xtr[tr_idx]), torch.tensor(ytr[tr_idx]); Xva, Yva = torch.tensor(xtr[val_idx]), torch.tensor(ytr[val_idx])
    Xte, Yte = torch.tensor(xte), torch.tensor(yte)
    routing = {'adaptive_girl': 'cos', 'adaptive_phase': 'phase'}.get(a.config)
    model = Net(routing)
    clusters = [list(range(10 * j, 10 * (j + 1))) for j in range(4)]; elementary = list(range(40, 64))
    reg = None
    if a.config != 'baseline':
        reg = CompositeRegulariser(tuple(model.dense.weight.shape), clusters, elementary, adaptive=a.config != 'semicomp')
        reg.init_from(model.dense.weight.detach())
    params = list(model.parameters()) + (list(p for p in reg.parameters() if p.requires_grad) if reg else [])
    opt = torch.optim.Adam(params, 1e-3)
    tag = f'cnn_{a.config}_g{a.gamma:g}_s{a.seed}' + ('_quick' if a.quick else '')
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    log_path = os.path.join(HERE, 'results', tag + '.log'); logf = open(log_path, 'a')
    def log(s):
        line = f'[{time.strftime("%H:%M:%S")}] {tag} {s}'; print(line, flush=True); logf.write(line + '\n'); logf.flush()
    log(f'config={a.config} gamma={a.gamma} seed={a.seed} epochs={a.epochs} n_train={len(Xtr)} n_val={len(Xva)} n_test={len(Xte)} params={sum(p.numel() for p in model.parameters())} torch={torch.__version__}')
    bs = 64; hist = []; t0 = time.time(); n_batches = (len(Xtr) + bs - 1) // bs
    for ep in range(a.epochs):
        model.train(); order = torch.tensor(rng.permutation(len(Xtr))); tot_loss = tot_pen = 0.0; correct = 0
        for b in range(n_batches):
            idx = order[b * bs:(b + 1) * bs]; xb, yb = Xtr[idx], Ytr[idx]
            logits = model(xb); loss = F.cross_entropy(logits, yb)
            pen = reg.penalty(model.dense.weight) if reg else torch.zeros(())
            total = loss + a.gamma * pen
            opt.zero_grad(); total.backward(); opt.step()
            tot_loss += loss.item() * len(idx); tot_pen += pen.item() * len(idx); correct += (logits.argmax(1) == yb).sum().item()
        model.eval(); va = accuracy(model, Xva, Yva)
        el = time.time() - t0
        rec = {'epoch': ep + 1, 'train_loss': tot_loss / len(Xtr), 'penalty': tot_pen / len(Xtr), 'train_acc': correct / len(Xtr), 'val_acc': va, 'elapsed_s': el}
        hist.append(rec); log(f'epoch {ep+1}/{a.epochs} loss {rec["train_loss"]:.4f} pen {rec["penalty"]:.3f} train {rec["train_acc"]:.4f} val {va:.4f} | {el/60:.1f} min, ETA {el/(ep+1)*(a.epochs-ep-1)/60:.1f} min')
    res = {'config': a.config, 'gamma': a.gamma, 'seed': a.seed, 'epochs': a.epochs, 'eps': a.eps, 'quick': a.quick, 'history': hist,
           'clean_test_acc': accuracy(model, Xte, Yte), 'val_acc_final': hist[-1]['val_acc']}
    log(f'clean test {res["clean_test_acc"]:.4f}'); t1 = time.time()
    Xad, Yad = Xte[:a.n_adv], Yte[:a.n_adv]; res['n_adv'] = len(Xad); res['clean_acc_on_adv_subset'] = accuracy(model, Xad, Yad)
    res['fgsm_test_acc'] = accuracy(model, Xad, Yad, attack=fgsm, eps=a.eps); log(f'FGSM eps={a.eps} on {len(Xad)} test images {res["fgsm_test_acc"]:.4f} ({time.time()-t1:.0f} s)'); t1 = time.time()
    torch.manual_seed(1000 + a.seed)
    res['pgd10_test_acc'] = accuracy(model, Xad, Yad, attack=pgd, eps=a.eps, steps=10); log(f'PGD-10 eps={a.eps} on {len(Xad)} test images {res["pgd10_test_acc"]:.4f} ({time.time()-t1:.0f} s)')
    res['cluster_compactness'] = cluster_compactness(model.dense.weight.detach(), clusters)
    if reg: res['regulariser_stats'] = reg.stats(model.dense.weight.detach()); res['final_penalty'] = float(reg.penalty(model.dense.weight.detach()))
    res['wall_time_s'] = time.time() - t0; res['n_params'] = sum(p.numel() for p in model.parameters())
    res['code_sha256_16'] = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:16]
    json.dump(res, open(os.path.join(HERE, 'results', tag + '.json'), 'w'), indent=1)
    log(f'compactness {res["cluster_compactness"]:.4f}; DONE in {res["wall_time_s"]/60:.1f} min')

if __name__ == '__main__':
    main()
