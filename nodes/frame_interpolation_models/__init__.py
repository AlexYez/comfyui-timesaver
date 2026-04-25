"""Frame interpolation model helpers for Timesaver nodes."""

from .film_net import FILMNet
from .ifnet import IFNet, detect_rife_config

__all__ = ["FILMNet", "IFNet", "detect_rife_config"]
