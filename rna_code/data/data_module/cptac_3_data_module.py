"""Module for handling CPTAC-3 datasets (train/test/val/whole)"""

from pathlib import Path

from .data_module import DataModuleABC


class CPTAC3DataModule(DataModuleABC):
    """Utility class to manage train/test data for the CPTAC-3 dataset"""

    def _pre_setup(self):
        self.dataset_type: str = "CPTAC-3"
        data_dir = self.data_param.get("Path", None)

        if data_dir is not None:
            self.build_from_scratch_flag = False
            self.default_data_path: Path = data_dir / "CPTAC_3_data.csv"
            self.default_metadata_path: Path = data_dir / "CPTAC_3_meta_data.csv"
