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
    dynamic_params (dict): Dictionary of lists of parameters that should be varied. 
                           Each key is a parameter name, and each value is a list of values to try for that parameter.

    Returns:
    list: A list of configuration dictionaries.
    """
    configurations = []

    # Extract the list of parameter values from the dynamic_params dictionary
    # and create a Cartesian product of all combinations (an N-dimensional grid)
    keys, values = zip(*dynamic_params.items())
    for combination in itertools.product(*values):
        # Merge dynamic parameters with their values for this combination
        dynamic_combination = dict(zip(keys, combination))

        # Create a new configuration by merging static and dynamic parameters
        config = {**static_params, **dynamic_combination}

        # Add this configuration to the list
        configurations.append(config)

    return configurations

def find_primes(n):
    primes = []
    for i in range(1,n+1):
        if n % i == 0:
            primes.append(i)
    return primes