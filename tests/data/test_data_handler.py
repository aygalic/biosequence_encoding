import unittest
from rna_code.data import data_handler

class TestExperiment(unittest.TestCase):
    
    def test_generate_dataset(self):
        data_handler.generate_dataset(
            dataset_type = "BRCA"
        )
        assert 1



if __name__ == "__main__":
    
    unittest.main()
