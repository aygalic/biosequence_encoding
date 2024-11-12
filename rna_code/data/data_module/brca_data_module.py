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

from pathlib import Path
from .data_module import DataModuleABC

class BRCADataModule(DataModuleABC):
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

        self.dataset_type : str = "BRCA"

        data_dir = self.data_param.get("Path", None)

        if data_dir is not None:
            self.default_data_path: Path = data_dir / "BRCA_data.csv"
            self.default_metadata_path: Path = data_dir / "meta_data.csv"