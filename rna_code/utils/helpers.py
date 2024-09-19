"""
Helpers Module

This module provides a collection of utility functions and classes to assist in various tasks 
related to Pytorch-based machine learning experiments. 

It includes tools for dataset handling, data formatting, encoding and reconstructing datasets, 
experiment configuration generation, and finding prime numbers. These utilities are designed to 
streamline the process of preparing data for model training and evaluation, as well as facilitating 
the experimentation with different model configurations.

Functions:
- format_dataset: Formats a given dataset for training with a PyTorch model.
- encode_recon_dataset: Encodes and reconstructs a dataset using a provided model.
- generate_config: Generates a list of configurations for machine learning experiments.
- find_primes: Finds prime numbers up to a given number.

Classes:
- Mydatasets: A custom PyTorch Dataset class for handling datasets.

The module primarily supports tasks in data preprocessing and experiment setup in a machine learning context.
"""

#from .. import config

import torch
import numpy as np

import itertools



class Mydatasets(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class for handling datasets.

    This class is designed to work with datasets in a format suitable for PyTorch models, 
    allowing for transformations and easy integration with PyTorch DataLoader.

    Args:
        data1: The primary dataset.
        transform (optional): A function/transform that takes in a sample and returns a transformed version.
    """
    def __init__(self, data1 ,transform = None):
        self.transform = transform
        self.data1 = data1
        self.datanum = len(data1)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        
        out_data1 = torch.tensor(self.data1[idx]).float() 
        if self.transform:
            out_data1 = self.transform(out_data1)

        return out_data1

def format_dataset(data):
    """
    Formats a given dataset for training with a PyTorch model, providing a dataloader object.

    Args:
        data: The dataset to be formatted.
    Returns:
        Tuple: A tuple containing the formatted dataset and the corresponding DataLoader.
    """
    print(data.shape)
    feature_num = data.shape[1]
    data = data.reshape(-1,1,feature_num)
    print(data.shape)

    batch_size = 32 # was 32 originally

    print('train data:',len(data))
    data_set = Mydatasets(data1 = data)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size = batch_size, shuffle=False)

    return data_set, dataloader

def encode_recon_dataset(dataloader, model, DEVICE):
    """
    Encodes and reconstructs a dataset using a provided model.

    Args:
        dataloader: The DataLoader containing the dataset to be processed.
        model: The model to be used for encoding and reconstruction.
        DEVICE: The device (CPU/GPU) on which the model is running.

    Returns:
        Tuple: A tuple containing the encoded and reconstructed outputs of the dataset.
    """
    en_lat = []
    en_reconstruction = []
    model = model.to(DEVICE)
    model.eval()
    #breakpoint()
    for inputs, _ in dataloader:
        latent = model.encode(inputs.to(DEVICE))
        data_recon = model(inputs.to(DEVICE))
        for elem in latent.cpu().detach().numpy():
            en_lat.append(elem)
        for elem in data_recon.cpu().detach().numpy():
            en_reconstruction.append(elem)
    encode_out = np.array(en_lat)
    reconstruction_out = np.array(en_reconstruction)
    return encode_out, reconstruction_out


def generate_config(static_params, dynamic_params):
    """
    Generate a list of configurations for ML experiments, combining static and dynamic parameters.

    Args:
    static_params (dict): Parameters that stay the same for each configuration.
    dynamic_params (dict): Dictionary of parameter names to lists of possible values. 
                           This includes both single-value parameters and coupled parameters (as tuples).

    Returns:
    list: A list of configuration dictionaries.
    """
    configurations = []

    # Prepare keys and respective values from dynamic parameters
    keys = list(dynamic_params.keys())
    values = list(dynamic_params.values())

    # Here, we recognize coupled parameters by checking if we have a tuple
    # We will convert tuples to dictionaries in the configurations
    for combination in itertools.product(*values):
        temp_config = dict(zip(keys, combination))

        # Flatten the configuration: if a value is a tuple, we unpack it
        # assuming it represents coupled parameters
        flat_config = {}
        for k, v in temp_config.items():
            if isinstance(v, tuple):
                # If the tuple is not directly the values but rather a pair of key-value itself
                # then we assume it's meant to unpack into the config as key-value pairs
                if all(isinstance(i, tuple) for i in v):
                    for inner_key, inner_value in v:
                        flat_config[inner_key] = inner_value
                else:
                    # if it's a simple tuple, just add it plainly
                    flat_config[k] = v
            else:
                flat_config[k] = v

        # Combine static and dynamic parameters into a single config dictionary
        config = {**static_params, **flat_config}
        configurations.append(config)

    return configurations

