"""
MNIST Autoencoder Package

A modern Python package for training autoencoders on MNIST digit data.
"""

__version__ = "0.1.0"
__author__ = "SAS Institute"

# Make submodules available at package level
from . import data
from . import models
from . import training
from . import evaluation

__all__ = ["data", "models", "training", "evaluation"]
