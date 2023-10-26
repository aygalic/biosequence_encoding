from .. import config
from .helpers import Mydatasets, encode_recon_dataset
from .visualisation import callback_viz

import math
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class Monitor():
    def __init__(self, model, dataloader, label, verbose = 1):
        self.checkpoints = [math.floor(x) for x in np.logspace(1,4)]
        self.feature_num = None
        self.DEVICE = torch.device(config["DEVICE"])
        self.train_res_recon_error = []
        self.perplexities = []
        self.frames = []
        self.model = model
        self.dataloader = dataloader
        self.label = label
        self.verbose = verbose
        
    def set_device(self, device):
        self.DEVICE = device
    def append_loss(self, value):
        self.train_res_recon_error.append(value)

    def callbacks(self, epoch):
        if (epoch + 1) in self.checkpoints:
            if(self.feature_num is None):
                self.feature_num = self.model.input_shape


            self.model.eval()

            encode_out, _ = encode_recon_dataset(self.dataloader, self.model, self.DEVICE)


            
            # PCA of the latent space
            pca = PCA(n_components=2)
            pca.fit(encode_out)
            pca_result = pca.transform(encode_out)


            index_column = np.full((pca_result.shape[0], 1), len(self.frames), dtype=int)

            pca_result_with_index = np.hstack((index_column, pca_result))

            self.frames.append(pca_result_with_index)

            if (epoch + 1) in self.checkpoints:

                x = iter(self.dataloader).__next__()

                if self.model.variational == "VAE":
                     x_reconstructed, _, _ = self.model.forward(x.to(self.DEVICE))
                else:
                    x_reconstructed = self.model.forward(x.to(self.DEVICE))

                x_reconstructed = x_reconstructed.cpu().detach().numpy()

                # stacking a single observation as well as its reconstruction in order to evaluate the results
                stack = np.vstack([x[0], x_reconstructed[0]])
                
                if(self.verbose >=1):
                    callback_viz(pca_result, encode_out, stack, self.train_res_recon_error, self.label)

            self.model.train()