"""Training system: framework-independent trainer contracts and the function trainer.

Heavy framework adapters (PyTorch Lightning, HF Trainer, JAX) plug in through
the same :class:`Trainer` protocol via the plugin system — core never imports
torch/tensorflow/jax.
"""

from aire.training.trainer import (
    Checkpoint,
    FunctionTrainer,
    Trainer,
    TrainingConfig,
    TrainResult,
)

__all__ = [
    "Checkpoint",
    "FunctionTrainer",
    "TrainResult",
    "Trainer",
    "TrainingConfig",
]
