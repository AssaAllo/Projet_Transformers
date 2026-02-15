"""
Projet G13 - Fine-tuning TinyBERT pour Emotion Detection
Package source avec tous les modules nécessaires
"""

__version__ = "1.0.0"
__author__ = "G13 Team"

from . import data_loader
from . import model_setup
from . import train_eval
from . import utils
from . import loss_landscape

__all__ = [
    'data_loader',
    'model_setup',
    'train_eval',
    'utils',
    'loss_landscape'
]