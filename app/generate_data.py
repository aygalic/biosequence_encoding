"""Generate data as a csv file"""

import pandas as pd

from rna_code.data import data_handler

print("running")

data_array, meta_data = data_handler.generate_dataset(
            dataset_type = "BRCA"
        )

dict_keys(['name', 'feature_names', 'seq_names', 'n_features', 'n_seq', 'meta_data', 'subtypes'])



meta_data["name"]
meta_data["feature_names"]
meta_data["seq_names"]
meta_data["seq_names"][0]
meta_data["n_features"]
meta_data["n_seq"]
meta_data["meta_data"]
type(meta_data["meta_data"])
type(meta_data["meta_data"][0])
meta_data["subtypes"]
pd.DataFrame(data_array)
breakpoint()