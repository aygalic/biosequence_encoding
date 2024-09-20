import numpy as np
import itertools
import torch

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
    for inputs, _ in dataloader:
        with torch.no_grad():
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

