import torch
import os
from pathlib import Path

HOME_PATH = Path(__file__).resolve().parent

DEVICE = torch.device("mps")
LOGFILE = HOME_PATH / "log.txt"
CACHE_PATH = HOME_PATH / "cache"