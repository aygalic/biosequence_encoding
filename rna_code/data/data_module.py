"""Module for handling BRCA datasets (train/test/val/whole)

Returns
-------
DataLoader
    DataLoader containing each subset
pd.DataFrame
    DataFrame containing metadata subsets

Raises
------
NotImplementedError
    Test set is not implemented yet.
"""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from rna_code import CACHE_PATH

from .dataset_builder import DatasetBuilder


class DataModule(pl.LightningDataModule):
    """Utility class to manage train/test data for the BRCA dataset

    Parameters
    ----------
    data_param : dict | None, optional
        Data parameters, can be either a dictionary containing a Path, or the
        parameters to build the dataset from scratch, by default None
    batch_size : int, optional
        Batch size, by default 32
    val_split : float, optional
        Portion of the dataset allocated to validation data, by default 0.2
    """

    def __init__(
        self,
        data_param: dict | None = None,
        batch_size: int = 32,
        val_split: float = 0.2,
    ):
        super().__init__()

        self.data_param = [data_param, {"Path": CACHE_PATH / "data"}][
            data_param is None
        ]
        self.batch_size = batch_size
        self.val_split = val_split
        self.data_array: np.ndarray
        self.meta_data: pd.DataFrame

        self.feature_num: int

        self.full_dataset: TensorDataset
        self.train_dataset: TensorDataset
        self.val_dataset: TensorDataset
        self.train_meta_data: pd.DataFrame
        self.val_meta_data: pd.DataFrame

    def setup(self, stage: str):
        """Set up the data module and pre-load everything.

        Parameters
        ----------
        stage : str
            See pytorch lightning documentation.
        """
        data_dir = self.data_param.get("Path", None)

        if data_dir is not None:
            data_path = data_dir / "BRCA_data.csv"
            metadata_path = data_dir / "meta_data.csv"
            self.data_array = pd.read_csv(data_path, index_col=0).values
            self.meta_data = pd.read_csv(metadata_path, index_col=0)

        else:
            builder = DatasetBuilder(dataset_type="BRCA")
            self.data_array, self.meta_data = builder.generate_dataset(
                **self.data_param
            )

        data_tensor = torch.from_numpy(self.data_array).float()

        self.feature_num = data_tensor.shape[1]

        # Create a dataset with both data and metadata
        self.full_dataset = TensorDataset(
            data_tensor, torch.arange(len(self.meta_data))
        )

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
        """Return the data loader for the entire dataset, pre split.

        Returns
        -------
        DataLoader
            Container for the whole dataset.
        """
        return DataLoader(self.full_dataset, batch_size=self.batch_size)

    @property
    def full_meta_data(self) -> pd.DataFrame:
        """Return the full metadata pre split.

        Returns
        -------
        pd.DataFrame
            Whole metadata
        """
        return self.meta_data

    def train_dataloader(self) -> DataLoader:
        """Training dataset.

        Returns
        -------
        DataLoader
            Container for the training set.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation Dataset.

        Returns
        -------
        DataLoader
            Container for the validation set.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Test Dataset

        Returns
        -------
        DataLoader
            If implemented, would return a container for the test set.

        Raises
        ------
        NotImplementedError
            No test set is implemented for the BRCA dataset as of right now.
        """
        raise NotImplementedError("No test set provided")
