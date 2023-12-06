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



