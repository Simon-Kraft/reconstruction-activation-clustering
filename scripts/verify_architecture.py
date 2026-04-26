"""
scripts/verify_architecture.py — Verify CNN architecture matches Table I.

Checks that the parameter counts in the paper match the actual model.

Usage:
    python scripts/verify_architecture.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.cnn import PaperCNN

# ── Expected values from Table I ─────────────────────────────────────
EXPECTED = {
    'conv1': {'params': 320,     'output': (32, 14, 14)},
    'conv2': {'params': 18_496,  'output': (64, 7, 7)},
    'fc1':   {'params': 401_536, 'output': (128,)},
    'fc2':   {'params': 1_290,   'output': (10,)},
}
EXPECTED_TOTAL = sum(v['params'] for v in EXPECTED.values())  # 421,642

def count_params(module):
    return sum(p.numel() for p in module.parameters())

def check_architecture():
    # Build model for MNIST (1×28×28 input)
    model = PaperCNN(n_channels=1, n_classes=10)
    model.eval()

    print("=" * 60)
    print("  CNN Architecture Verification")
    print("  Input: 1×28×28 (MNIST / FashionMNIST)")
    print("=" * 60)

    # ── Run a forward pass to get output shapes ───────────────────────
    dummy = torch.zeros(1, 1, 28, 28)
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.shape
        return hook

    model.conv1.register_forward_hook(make_hook('after_conv1'))
    model.pool.register_forward_hook(make_hook('after_pool1'))
    # We need shapes after each conv+relu+pool block
    # Use a manual forward pass instead

    with torch.no_grad():
        x = dummy
        # conv1 block
        x = model.conv1(x)        # Conv2d
        x = torch.relu(x)
        x = model.pool(x)         # MaxPool
        shape_conv1 = tuple(x.shape[1:])  # drop batch dim

        # conv2 block
        x = model.conv2(x)
        x = torch.relu(x)
        x = model.pool(x)
        shape_conv2 = tuple(x.shape[1:])

        # flatten
        x = x.view(x.size(0), -1)

        # fc1
        x = model.fc1(x)
        shape_fc1 = tuple(x.shape[1:])
        x = torch.relu(x)

        # fc2
        x = model.fc2(x)
        shape_fc2 = tuple(x.shape[1:])

    shapes = {
        'conv1': shape_conv1,
        'conv2': shape_conv2,
        'fc1':   shape_fc1,
        'fc2':   shape_fc2,
    }

    # ── Count parameters per layer ────────────────────────────────────
    layer_modules = {
        'conv1': model.conv1,
        'conv2': model.conv2,
        'fc1':   model.fc1,
        'fc2':   model.fc2,
    }

    all_ok = True
    print(f"\n  {'Layer':<8} {'Params (actual)':>16} {'Params (paper)':>16} "
          f"{'Shape (actual)':>18} {'Shape (paper)':>18} {'OK?':>6}")
    print("  " + "-" * 86)

    for name in ['conv1', 'conv2', 'fc1', 'fc2']:
        actual_params = count_params(layer_modules[name])
        expected_params = EXPECTED[name]['params']
        actual_shape = shapes[name]
        expected_shape = EXPECTED[name]['output']

        params_ok = actual_params == expected_params
        shape_ok  = actual_shape  == expected_shape
        ok = params_ok and shape_ok
        if not ok:
            all_ok = False

        status = "✓" if ok else "✗ MISMATCH"
        print(f"  {name:<8} {actual_params:>16,} {expected_params:>16,} "
              f"{str(actual_shape):>18} {str(expected_shape):>18} {status:>6}")

    # ── Total ─────────────────────────────────────────────────────────
    total_actual = count_params(model)
    total_expected = EXPECTED_TOTAL
    total_ok = total_actual == total_expected

    print("  " + "-" * 86)
    status = "✓" if total_ok else "✗ MISMATCH"
    print(f"  {'TOTAL':<8} {total_actual:>16,} {total_expected:>16,} "
          f"{'':>18} {'':>18} {status:>6}")

    # ── Parameter breakdown for conv layers ──────────────────────────
    print("\n  Parameter breakdown:")
    print(f"  conv1: {model.conv1.weight.numel()} weights "
          f"+ {model.conv1.bias.numel()} bias "
          f"= {count_params(model.conv1)}")
    print(f"    weight shape: {tuple(model.conv1.weight.shape)}  "
          f"(out_ch × in_ch × kH × kW = "
          f"{model.conv1.out_channels} × {model.conv1.in_channels} × "
          f"{model.conv1.kernel_size[0]} × {model.conv1.kernel_size[1]})")

    print(f"  conv2: {model.conv2.weight.numel()} weights "
          f"+ {model.conv2.bias.numel()} bias "
          f"= {count_params(model.conv2)}")
    print(f"    weight shape: {tuple(model.conv2.weight.shape)}  "
          f"(out_ch × in_ch × kH × kW = "
          f"{model.conv2.out_channels} × {model.conv2.in_channels} × "
          f"{model.conv2.kernel_size[0]} × {model.conv2.kernel_size[1]})")

    print(f"  fc1:   {model.fc1.weight.numel()} weights "
          f"+ {model.fc1.bias.numel()} bias "
          f"= {count_params(model.fc1)}")
    print(f"    weight shape: {tuple(model.fc1.weight.shape)}  "
          f"(out × in = {model.fc1.out_features} × {model.fc1.in_features})")

    print(f"  fc2:   {model.fc2.weight.numel()} weights "
          f"+ {model.fc2.bias.numel()} bias "
          f"= {count_params(model.fc2)}")
    print(f"    weight shape: {tuple(model.fc2.weight.shape)}  "
          f"(out × in = {model.fc2.out_features} × {model.fc2.in_features})")

    print()
    if all_ok and total_ok:
        print("  ✓ All parameter counts and output shapes match Table I.")
    else:
        print("  ✗ Mismatches found — check your paper table.")

    return all_ok and total_ok


if __name__ == '__main__':
    ok = check_architecture()
    sys.exit(0 if ok else 1)