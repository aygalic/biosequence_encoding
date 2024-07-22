import torch
import os
from pathlib import Path

HOME_PATH = Path(__file__).resolve().parent

DEVICE = torch.device("mps")