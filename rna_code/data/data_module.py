import pickle
from pathlib import Path
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from rna_code import CACHE_PATH
import pandas as pd


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: Path = CACHE_PATH / "data",
            batch_size: int = 32,
            val_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage: str):
        data_path = self.data_dir / 'BRCA_data.csv'
        metadata_path = self.data_dir / 'meta_data.csv'

        data_array = pd.read_csv(data_path, index_col=0).values
        self.meta_data = pd.read_csv(metadata_path, index_col=0)

        # Convert NumPy array to PyTorch tensor
        data_tensor = torch.from_numpy(data_array).float()
        
        feature_num = data_tensor.shape[1]
        data_tensor = data_tensor.reshape(-1, 1, feature_num)
        
        # Create a dataset with both data and metadata
        full_dataset = TensorDataset(data_tensor, torch.arange(len(self.meta_data)))
        
        # Split dataset
        dataset_size = len(full_dataset)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Split meta_data based on the indices
        train_indices = self.train_dataset.indices
        val_indices = self.val_dataset.indices
        
        self.train_meta_data = self.meta_data.iloc[train_indices]
        self.val_meta_data = self.meta_data.iloc[val_indices]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)