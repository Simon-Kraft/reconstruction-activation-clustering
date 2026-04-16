import sys
sys.path.insert(0, '..')

import torch
from models.cnn import PaperCNN
from data.loader import load_dataset

dataset_info = load_dataset('MNIST', data_dir='../datasets/')
model        = PaperCNN.for_dataset(dataset_info)
model.eval()

# Run a dummy forward pass and check output shapes via hooks
dummy = torch.zeros(1, 1, 28, 28)

print(f"{'Layer':<8} {'Type':<25} {'Output shape':<20} {'Parameters':>12}")
print("-" * 70)

layers = [
    ('conv1', model.conv1),
    ('conv2', model.conv2),
    ('fc1',   model.fc1),
    ('fc2',   model.fc2),
]

for name, layer in layers:
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"{name:<8} {str(type(layer).__name__):<25} {'':20} {n_params:>12,}")

print()

# Also run forward pass to get actual output shapes
shapes = {}
hooks  = []

def make_hook(n):
    def hook(module, input, output):
        shapes[n] = tuple(output.shape[1:])
    return hook

for name, layer in layers:
    hooks.append(layer.register_forward_hook(make_hook(name)))

with torch.no_grad():
    model(dummy)

for h in hooks:
    h.remove()

print(f"{'Layer':<8} {'Output shape':<20}")
print("-" * 30)
for name, _ in layers:
    print(f"{name:<8} {str(shapes[name]):<20}")

print()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")