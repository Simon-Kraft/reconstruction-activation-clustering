import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseACModel(nn.Module):
    """Parent class to handle activation hooking logic for any architecture."""
    def __init__(self, activation='relu'):
        super().__init__()
        
        self.activation_fn = None
        if activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        else:
            self.activation_fn = nn.ReLU()
        
        self._activations = {}
        self._hooks = []

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self._activations[name] = output.detach().cpu()
        return hook_fn

    def _register_hooks(self, layers_dict):
        for name, module in layers_dict.items():
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_activations(self):
        return {k: v.clone() for k, v in self._activations.items()}


class LargeCNN(BaseACModel):
    def __init__(self, activation='relu'):
        super(LargeCNN, self).__init__(activation)
        
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32,  kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
        self.LAYER_REGISTRY = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'fc1':   self.fc1,
            'fc2':   self.fc2,
            'fc3':   self.fc3,
        }
        
        self.LAYER_META = {
            'conv1': {'type': 'Conv',  'depth': 1, 'channels': 32,  'spatial': '14×14'},
            'conv2': {'type': 'Conv',  'depth': 2, 'channels': 64,  'spatial': '7×7'},
            'conv3': {'type': 'Conv',  'depth': 3, 'channels': 128, 'spatial': '7×7'},
            'fc1':   {'type': 'FC',    'depth': 4, 'channels': 256, 'spatial': '—'},
            'fc2':   {'type': 'FC',    'depth': 5, 'channels': 128, 'spatial': '—'},
            'fc3':   {'type': 'FC',    'depth': 6, 'channels': 10,  'spatial': '—'},
        }
        
        self._register_hooks(self.LAYER_REGISTRY)
    
    def forward(self, x):
        x = self.activation_fn(self.conv1(x))
        x = self.activation_fn(self.conv2(x))
        x = self.activation_fn(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation_fn(self.fc1(self.dropout(x)))
        x = self.activation_fn(self.fc2(self.dropout(x)))
        return self.fc3(x)


class MidCNN(BaseACModel):
    def __init__(self, activation='sigmoid'):
        super(MidCNN, self).__init__(activation)
        self.conv1 = nn.Conv2d(1,  12, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=5, stride=1, padding=2)
        
        self.fc = nn.Linear(12 * 7 * 7, 10)

        self.LAYER_REGISTRY = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'fc':    self.fc,
        }
        
        self.LAYER_META = {
            'conv1': {'type': 'Conv', 'depth': 1, 'channels': 12, 'spatial': '14×14'},
            'conv2': {'type': 'Conv', 'depth': 2, 'channels': 12, 'spatial': '7×7'},
            'conv3': {'type': 'Conv', 'depth': 3, 'channels': 12, 'spatial': '7×7'},
            'fc':    {'type': 'FC',   'depth': 4, 'channels': 10, 'spatial': '—'},
        }
        
        self._register_hooks(self.LAYER_REGISTRY)

    def forward(self, x):
        x = self.activation_fn(self.conv1(x))
        x = self.activation_fn(self.conv2(x))
        x = self.activation_fn(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SmallCNN(BaseACModel):
    def __init__(self, activation='relu'):
        super(SmallCNN, self).__init__(activation)
        
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
        self.LAYER_REGISTRY = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'fc':    self.fc,
        }
        
        self.LAYER_META = {
            'conv1': {'type': 'Conv', 'depth': 1, 'channels': 16, 'spatial': '14×14'},
            'conv2': {'type': 'Conv', 'depth': 2, 'channels': 32, 'spatial': '7×7'},
            'fc':    {'type': 'FC',   'depth': 3, 'channels': 10, 'spatial': '—'},
        }
        
        self._register_hooks(self.LAYER_REGISTRY)

    def forward(self, x):
        x = self.activation_fn(self.conv1(x))
        x = self.activation_fn(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)