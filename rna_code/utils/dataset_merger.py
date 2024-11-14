import pandas as pd

class DatasetMerger():

    @staticmethod
    def intersect(dataset1 : pd.DataFrame, dataset2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_intersection = set(dataset1.columns) & set(dataset2.columns)
        return pd.concat(dataset1[feature_intersection], dataset2[feature_intersection], axis = 1)

    @staticmethod
    def union(dataset1 : pd.DataFrame, dataset2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass