import os
import sys
import pandas as pd


#sys.path.append('../')
from rna_code.data import data_handler


pd.options.display.width = 1000

def run_BRCA_analysis():

    # processed data

    data, metadata = data_handler.generate_dataset(
        dataset_type="BRCA",
        LS_threshold= 0.0020,
        MAD_threshold = 1, 
        #MT_removal= True, 
        expression_threshold= 0.1, keep_only_protein_coding = False, verbose = 1)


if __name__ == "__main__":
    # Your code here
    run_BRCA_analysis()