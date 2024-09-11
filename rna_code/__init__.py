import torch
import os
from pathlib import Path

HOME_PATH = Path(__file__).resolve().parent
OUTPUT_PATH = HOME_PATH / '..' / "outputs"
STATIC_OUTPUT_PATH = HOME_PATH / '..' / "static"

DEVICE = torch.device("mps")
LOGFILE = OUTPUT_PATH / "log.txt"
CACHE_PATH = HOME_PATH / "cache"
DOCS_PATH = HOME_PATH / "../docs"
