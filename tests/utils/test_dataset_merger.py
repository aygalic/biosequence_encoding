import unittest
import pandas as pd
from rna_code.utils.dataset_merger import DatasetMerger

class TestDatasetMerger(unittest.TestCase):
    def test_intersect_dataset(self):

        dataset_1 = pd.DataFrame(data = {
            "a" : [1,2],
            "b" : [3,4]
        })
        dataset_2 = pd.DataFrame(data = {
            "a" : [1,2],
            "c" : [3,4]
        })
        merged_df = DatasetMerger.intersect(dataset_1, dataset_2)
        pd.testing.assert_frame_equal(merged_df, pd.DataFrame(data={"a": [1,2,1,2]}, index = [0,1,0,1]))

    def test_union_dataset(self):

        dataset_1 = pd.DataFrame(data = {
            "a" : [1,2],
            "b" : [3,4]
        })
        dataset_2 = pd.DataFrame(data = {
            "a" : [1,2],
            "c" : [3,4]
        })
        merged_df = DatasetMerger.union(dataset_1, dataset_2)
        pd.testing.assert_frame_equal(merged_df, pd.DataFrame(
            data={
                "a": [1,2,1,2],
                "b": [3,4,0,0],
                "c": [0,0,3,4]},
            index = [0,1,0,1]))



if __name__ == "__main__":
    unittest.main()
