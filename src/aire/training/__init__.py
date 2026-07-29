"""Training system: framework-independent trainer contracts and the function trainer.

Heavy framework adapters (PyTorch Lightning, HF Trainer, JAX) plug in through
the same :class:`Trainer` protocol via the plugin system — core never imports
torch/tensorflow/jax. Quantization and distillation adapters live alongside.
"""

from aire.training.distill import (
    DistillationConfig,
    Distiller,
    DistillTrainer,
    HFDistillTrainer,
    create_distiller,
    create_hf_distiller,
    soft_kl_loss,
)
from aire.training.foundation import (
    FoundationConfig,
    FoundationModel,
    create_foundation,
)
from aire.training.foundation import catalog as foundation_catalog
from aire.training.quantize import QuantizationConfig, Quantizer, create_quantizer
from aire.training.trainer import (
    Checkpoint,
    FunctionTrainer,
    Trainer,
    TrainingConfig,
    TrainResult,
)

__all__ = [
    "Checkpoint",
    "DistillTrainer",
    "DistillationConfig",
    "Distiller",
    "FoundationConfig",
    "FoundationModel",
    "FunctionTrainer",
    "HFDistillTrainer",
    "QuantizationConfig",
    "Quantizer",
    "TrainResult",
    "Trainer",
    "TrainingConfig",
    "create_distiller",
    "create_foundation",
    "create_hf_distiller",
    "create_quantizer",
    "foundation_catalog",
    "soft_kl_loss",
]
