from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from rna_code import CACHE_PATH

from .dataset_builder import DatasetBuilder


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_param: Path = {"Path" : CACHE_PATH / "data"},
            batch_size: int = 32,
            val_split: float = 0.2):
        super().__init__()

        self.data_param = data_param
        self.batch_size = batch_size
        self.val_split = val_split
        self.data_array : np.ndarray
        self.meta_data : pd.DataFrame
        self.feature_num : int

    def setup(self, stage: str):
        data_dir = self.data_param.get("Path", None)

        if data_dir is not None:
            data_path = data_dir / 'BRCA_data.csv'
            metadata_path = data_dir / 'meta_data.csv'
            self.data_array = pd.read_csv(data_path, index_col=0).values
            self.meta_data = pd.read_csv(metadata_path, index_col=0)

        else:
            builder = DatasetBuilder(dataset_type="BRCA")
            self.data_array, self.meta_data = builder.generate_dataset(**self.data_param)

        data_tensor = torch.from_numpy(self.data_array).float()

        self.feature_num = data_tensor.shape[1]

        # Create a dataset with both data and metadata
        self.full_dataset = TensorDataset(data_tensor, torch.arange(len(self.meta_data)))

        # Split dataset
        dataset_size = len(self.full_dataset)
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size]
        )

        # Split meta_data based on the indices
        train_indices = self.train_dataset.indices
        val_indices = self.val_dataset.indices

        self.train_meta_data = self.meta_data.iloc[train_indices]
        self.val_meta_data = self.meta_data.iloc[val_indices]


    def full_data_loader(self) -> DataLoader:
        return DataLoader(self.full_dataset, batch_size=self.batch_size)

    @property
    def full_meta_data(self) -> pd.DataFrame:
        return self.meta_data

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("No test set provided")
