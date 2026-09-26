#!/usr/bin/env python3
"""Verify that the local CIFAR-10 input is the archived public data set."""
from pathlib import Path
import hashlib, json, os, sys

HERE = Path(__file__).resolve().parent
M = json.load(open(HERE / 'cifar10_input_manifest.json'))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

roots = [HERE / 'data' / 'cifar-10-batches-py',
         Path(os.path.expanduser('~/.keras/datasets/cifar-10-batches-py'))]
root = next((p for p in roots if p.is_dir()), None)
if root is None:
    print('CIFAR10_INPUT_FAIL: data directory not found')
    print('Download source:', M['source_url'])
    sys.exit(1)

bad = []
for name, expected in M['files_sha256'].items():
    p = root / name
    got = sha256(p) if p.is_file() else None
    ok = got == expected
    print(('PASS' if ok else 'FAIL'), name, got)
    if not ok: bad.append(name)
print('CIFAR10_INPUT_'+('PASS' if not bad else 'FAIL'), root)
sys.exit(1 if bad else 0)
