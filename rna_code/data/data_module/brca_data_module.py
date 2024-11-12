"""Module for handling BRCA datasets (train/test/val/whole)"""

from pathlib import Path

from .data_module import DataModuleABC


class BRCADataModule(DataModuleABC):
    """Utility class to manage train/test data for the BRCA dataset"""

    def _pre_setup(self):
        self.dataset_type: str = "BRCA"
        data_dir = self.data_param.get("Path", None)

        if data_dir is not None:
            self.build_from_scratch_flag = False
            self.default_data_path: Path = data_dir / "BRCA_data.csv"
            self.default_metadata_path: Path = data_dir / "meta_data.csv"
