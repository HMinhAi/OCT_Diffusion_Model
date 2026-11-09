"""Logger module for PatchCL-AE training.

Provides structured logging to:
- metrics_history.csv
- metrics_history.json
- summary.json
- TensorBoard
"""

from __future__ import annotations

import sys
from pathlib import Path

# Try importing from utils.training_logger if in parent repo
try:
    from utils.training_logger import TrainingLogger
except ImportError:
    # If run in isolation, import directly or resolve parent path
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from utils.training_logger import TrainingLogger

__all__ = ["TrainingLogger"]
