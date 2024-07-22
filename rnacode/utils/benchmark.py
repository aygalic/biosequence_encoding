"""
This module provides utility functions for model analysis and data characterization.

Functions:

- `count_parameters(model)`: 
  Calculates and prints the total number of parameters and the number of trainable parameters in a given PyTorch model. This function is useful for understanding the complexity and capacity of the model.

  Args:
      model (torch.nn.Module): The PyTorch model for which parameters are counted.
  
  Returns:
      tuple: A tuple containing the total number of parameters and the number of trainable parameters in the model.

- `hopkins(data, n=None)`:
  Computes the Hopkins statistic for a given dataset to assess its cluster tendency or the likelihood of data containing meaningful clusters. A value close to 1 indicates high cluster tendency, while a value around 0.5 suggests random distribution.

  Args:
      data (numpy.ndarray): The dataset for which the Hopkins statistic is to be calculated.
      n (int, optional): The number of points to sample. If None, uses the length of the dataset.

  Returns:
      float: The Hopkins statistic for the dataset.

These functions are essential tools for model analysis, helping to assess model size and complexity, as well as evaluating the data's inherent clustering tendency.

Example Usage:
    model = SomePyTorchModel()
    total_params, trainable_params = count_parameters(model)
    hopkins_statistic = hopkins(some_dataset)
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors

def count_parameters(model):
    ''' for torch models, print number of param. '''
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return total_params, trainable_params


def hopkins(data, n=None):
    if n is None:
        n = len(data)
    
    # Step 1: Randomly sample n data points from data
    sample = data[np.random.choice(data.shape[0], n, replace=False)]
    
    # Step 2: Nearest neighbor distances for real data
    nbrs = NearestNeighbors(n_neighbors=2).fit(data)
    _, indices = nbrs.kneighbors(sample)
    w_distances = np.array([np.linalg.norm(sample[i] - data[indices[i, 1]]) for i in range(n)])
    
    # Step 3: Generate n random data points
    mins, maxs = np.min(data, axis=0), np.max(data, axis=0)
    random_data = np.array([np.random.uniform(mins[i], maxs[i], n) for i in range(data.shape[1])]).T
    
    # Step 4: Nearest neighbor distances for random data
    _, indices = nbrs.kneighbors(random_data)
    u_distances = np.array([np.linalg.norm(random_data[i] - data[indices[i, 0]]) for i in range(n)])
    
    # Step 5: Calculate Hopkins statistic
    H = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))
    return H



