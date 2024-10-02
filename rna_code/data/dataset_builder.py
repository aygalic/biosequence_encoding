"""Dataset Builder

For now, only supports BRCA data, but can be extended to support various dataset

Returns
-------
tuple[pd.DataFrame, pd.DataFrame]
    Dataset and associated metadata.

Raises
------
NotImplementedError
    If trying to build a dataset that is not implemented
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize

from rna_code.data.interface import base_interface

from .feature_selection.mad_selector import MADSelector
from .feature_selection.expression_selector import ExpressionSelector
from .feature_selection.laplacian_selector import LaplacianSelector

from .interface.brca_interface import BRCAInterface


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MAD_THRESHOLD = 1
DEFAULT_LS_THRESHOLD = 0.002
DEFAULT_EXPRESSION_THRESHOLD = None
DEFAULT_NORMALIZATION = False
DEFAULT_KEEP_ONLY_PROTEIN_CODING = False
DEFAULT_LOG1P = True
DEFAULT_MINMAX = True
DEFAULT_SORTING = False


class DatasetBuilder:
    """Builder class for dataset associated with metadata. Supervise steps from raw
    data into processing steps

    Parameters
    ----------
    dataset_type : Literal["BRCA"]
        Name of the dataset to build.
    selection_thresholds : dict[str, float], optional
        Dictionary containing all threshold used during feature selection,
        by default None
    additional_processing_steps : dict[str, bool], optional
        Dictionary containing all the additional processing steps, by default None

    Raises
    ------
    NotImplementedError
        If trying to build a non supported dataset.
    """
    def __init__(
        self,
        dataset_type: Literal["BRCA"],
        selection_thresholds: dict[str, float] = None,
        additional_processing_steps: dict[str, bool] = None,
    ) -> None:
        self.data_interface: base_interface
        match dataset_type:
            case "BRCA":
                self.data_interface = BRCAInterface()
            case _:
                raise NotImplementedError

        if selection_thresholds is None:
            selection_thresholds = {}
        self.mad_threshold = selection_thresholds.get(
            "MAD_threshold", DEFAULT_MAD_THRESHOLD
        )
        self.ls_threshold = selection_thresholds.get(
            "LS_threshold", DEFAULT_LS_THRESHOLD
        )
        self.expression_threshold = selection_thresholds.get(
            "expression_threshold", DEFAULT_EXPRESSION_THRESHOLD
        )

        if additional_processing_steps is None:
            additional_processing_steps = {}
        self.normalization = additional_processing_steps.get(
            "normalization", DEFAULT_NORMALIZATION
        )
        self.keep_only_protein_coding = additional_processing_steps.get(
            "keep_only_protein_coding", DEFAULT_KEEP_ONLY_PROTEIN_CODING
        )
        self.log1p = additional_processing_steps.get("log1p", DEFAULT_LOG1P)
        self.min_max = additional_processing_steps.get("min_max", DEFAULT_MINMAX)
        self.sort_symbols = additional_processing_steps.get(
            "sort_symbols", DEFAULT_SORTING
        )

        self.data_array: np.ndarray
        self.names: pd.DataFrame
        self.meta_data: pd.DataFrame
        self.subtypes: list[str]
        self.entry_names: list[str]

    def _build_unprocessed_component(self) -> None:
        """Builds raw dataset"""
        self.data_interface.setup()
        self.data_array = self.data_interface.data_array
        self.names = self.data_interface.names
        self.meta_data = self.data_interface.meta_data
        self.subtypes = self.data_interface.subtypes
        self.entry_names = self.data_interface.entry_names

    def _feature_selection(self) -> None:
        """Perform feature selection according to selection_thresholds"""
        expression_selector = ExpressionSelector(threshold=self.expression_threshold)
        mad_selector = MADSelector(threshold=self.mad_threshold)
        laplacian_selector = LaplacianSelector(threshold=self.ls_threshold)

        if self.expression_threshold is not None:
            gene_selected = expression_selector.select_features(self.data_array)
            self.data_array = self.data_array[:, gene_selected]
            self.names = self.names[gene_selected]

        if self.mad_threshold is not None:
            gene_selected = mad_selector.select_features(self.data_array)
            self.data_array = self.data_array[:, gene_selected]
            self.names = self.names[gene_selected]

        if self.ls_threshold is not None:
            gene_selected = laplacian_selector.select_features(self.data_array)
            self.data_array = self.data_array[:, gene_selected]
            self.names = self.names[gene_selected]

        if self.keep_only_protein_coding:
            breakpoint()
            # FIXME this should be moved to dataset creation (BRCA specific)
            gene_selected = self.names["gene_type"] == "protein_coding"
            logging.info(
                "removing %i non coding genes from dataset",
                len(gene_selected) - sum(gene_selected),
            )
            self.data_array = self.data_array[:, gene_selected]
            self.names = self.names[gene_selected]

        logger.debug("number of genes selected : %i", len(self.data_array[0]))

    def _feature_transformation(self) -> None:
        """Perform feature transformation according to additional_processing_steps"""
        if self.sort_symbols:
            logging.info(
                "sorting based on genomic position chr then transcript start..."
            )
            self.names = self.names.sort_values(
                ["genomic_pos.chr", "genomic_pos.start"], ascending=[True, True]
            )
            self.data_array = self.data_array[:, self.names.index]

        if self.normalization:
            logging.info("normalizing data...")
            self.data_array = normalize(self.data_array)

        if self.log1p:
            logging.info("log(1 + x) transformation...")
            self.data_array = np.log1p(self.data_array)

        if self.min_max:
            logging.info("scaling to [0, 1]...")
            scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
            self.data_array = scaler.fit_transform(self.data_array)

    def generate_dataset(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate dataset, from data loading to feature selection and transformation.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing dataset and metadata.
        """
        self._build_unprocessed_component()
        self._feature_selection()
        self._feature_transformation()

        logging.info("number of seq in the dataset : %i", len(self.data_array))

        df = pd.DataFrame(
            data=self.data_array, index=self.entry_names, columns=self.names["gene_id"]
        )

        meta_dict = {
            "meta_data": self.meta_data,
            "subtypes": self.subtypes,
        }
        meta_df = pd.DataFrame(meta_dict, index=self.entry_names)

        return df, meta_df
