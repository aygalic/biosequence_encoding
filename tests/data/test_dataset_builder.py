import unittest
from rna_code.data.dataset_builder import DatasetBuilder

class TestExperiment(unittest.TestCase):
    
    def test_generate_dataset(self):
        builder = DatasetBuilder(dataset_type = "BRCA")
        builder.generate_dataset()
        assert 1

if __name__ == "__main__":    
    unittest.main()
