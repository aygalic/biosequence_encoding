"""Merge datasets"""
import pandas as pd

class DatasetMerger():
    """Merge pandas datasets using either feature intersection or union.

    Returns
    -------
    pd.DataFrame
        New Merged dataset
    """

    @staticmethod
    def intersect(dataset1 : pd.DataFrame, dataset2: pd.DataFrame) -> pd.DataFrame:
        """Merge two datasets by intersecting feature in common.
        Missing features from one dataset are discarded.

        Parameters
        ----------
        dataset1 : pd.DataFrame
            First dataset
        dataset2 : pd.DataFrame
            Second dataset

        Returns
        -------
        pd.DataFrame
            Merged dataset
        """
        feature_intersection = list(set(dataset1.columns) & set(dataset2.columns))
        return pd.concat([dataset1[feature_intersection], dataset2[feature_intersection]], axis = 0)

    @staticmethod
    def union(dataset1 : pd.DataFrame, dataset2: pd.DataFrame) -> pd.DataFrame:
        """Merge datasets by union. Missing features from one dataset are 0 padded.

        Parameters
        ----------
        dataset1 : pd.DataFrame
            First dataset
        dataset2 : pd.DataFrame
            Second Dataset

        Returns
        -------
        pd.DataFrame
            Merged dataset
        """
        feature_union = set(dataset1.columns)
        feature_union.update(set(dataset2.columns))
        
        new_df = pd.DataFrame(columns=sorted(list(feature_union)))
        return pd.concat([new_df, dataset1, dataset2], axis=0).fillna(0).astype(dataset1.dtypes[0])
