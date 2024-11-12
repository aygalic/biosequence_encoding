"""Module for handling CPTAC-3 datasets (train/test/val/whole)

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

class CPTAC3DataModule(DataModuleABC):
    """Utility class to manage train/test data for the CPTAC-3 dataset

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
    def _pre_setup(self):
        self.dataset_type : str = "CPTAC-3"
        data_dir = self.data_param.get("Path", None)

        if data_dir is not None:
            self.build_from_scratch_flag = False
            self.default_data_path: Path = data_dir / "CPTAC_3_data.csv"
            self.default_metadata_path: Path = data_dir / "CPTAC_3_meta_data.csv"