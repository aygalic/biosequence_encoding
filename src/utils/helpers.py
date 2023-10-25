from .. import config

import torch
import numpy as np

from sklearn.model_selection import train_test_split

import itertools



class Mydatasets(torch.utils.data.Dataset):
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

def format_dataset(data, metadata, test_size = 0):
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
    en_lat = []
    en_reconstruction = []

    model.eval()
    for _, inputs in enumerate(dataloader):
        if model.variational:
            latent_1, _ = model.encode(inputs.to(DEVICE))
            data_recon, _, _ = model(inputs.to(DEVICE))
        else:
            latent_1 = model.encode(inputs.to(DEVICE))
            data_recon = model(inputs.to(DEVICE))

        
        for elem in latent_1.cpu().detach().numpy():
            en_lat.append(elem)

        for elem in data_recon.cpu().detach().numpy():
            en_reconstruction.append(elem)

        
    encode_out = np.array(en_lat)
    if(not model.convolution):
        encode_out = encode_out.squeeze(axis=1)
    reconstruction_out = np.array(en_reconstruction).squeeze(axis=1)

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

def find_primes(n):
    primes = []
    for i in range(1,n+1):
        if n % i == 0:
            primes.append(i)
    return primes