import torch
import os
from pathlib import Path

HOME_PATH = Path(__file__).resolve().parent
OUTPUT_PATH = HOME_PATH / "outputs"

DEVICE = torch.device("mps")
LOGFILE = OUTPUT_PATH / "log.txt"
CACHE_PATH = HOME_PATH / "cache"