"""
data/ — Dataset loading, trigger injection, reconstruction, and dataset building.

Public API:

    from data.loader       import load_dataset, DatasetInfo
    from data.trigger      import TriggerConfig
    from data.reconstruction import ReconConfig, intercept_gradients, reconstruct
    from data.builder      import PoisonConfig, MixedDataset, build_poisoned_dataset
"""

from data.loader         import load_dataset, DatasetInfo
from data.trigger        import TriggerConfig
from data.reconstruction import ReconConfig, intercept_gradients, reconstruct
from data.builder        import PoisonConfig, MixedDataset, build_poisoned_dataset

__all__ = [
    "load_dataset",
    "DatasetInfo",
    "TriggerConfig",
    "ReconConfig",
    "intercept_gradients",
    "reconstruct",
    "PoisonConfig",
    "MixedDataset",
    "build_poisoned_dataset",
]