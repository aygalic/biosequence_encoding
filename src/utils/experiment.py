from .. import config

import sys
import pickle
import importlib

sys.path.append('..')
#from utils import data_handler
from src.utils import data_handler
from src import config
from src.utils import visualisation, benchmark, helpers, monitoring
from src.models import model


# data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data analysis
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# pytorch specific
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = torch.device(config["DEVICE"])
LOGFILE = config["LOGFILE"]



from sklearn import datasets
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def log_experiment(record, csv_path=LOGFILE):
    # Create a DataFrame from the new record
    new_df = pd.DataFrame([record])
    
    # Check if the CSV file exists
    try:
        # Read the existing data
        existing_df = pd.read_csv(csv_path)
        
        # Concatenate the new data with the existing data
        # This aligns data by columns names, inserting NaNs where columns do not match
        combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    except FileNotFoundError:
        # If the CSV doesn't exist, the new data is all we need to write
        combined_df = new_df
    
    # Write the combined DataFrame back to CSV
    combined_df.to_csv(csv_path, index=False)


class Experiment():
    def build_dataset(self, data_param):
        data_param["verbose"] = self.verbose - 1
        self.data, self.metadata = data_handler.generate_dataset_BRCA(**data_param)
        self.input_shape = len(self.metadata["feature_names"])
        print("input shape :", self.input_shape)

    def load_dataset(self, data_param):
        with open(data_param, 'rb') as f:
            self.data, self.metadata = pickle.load(f)
        self.input_shape = len(self.metadata["feature_names"])
        print("input shape :", self.input_shape)

    def build_model(self, shape, model_param):
        if "transformer" in model_param:
            if model_param["transformer"] == True:
                num_heads_candidate = helpers.find_primes(self.input_shape)
                if(len(num_heads_candidate) > 1):
                    self.model_param["num_heads"] = num_heads_candidate[-1]
                else:
                    self.model_param["num_heads"] = num_heads_candidate[-2]

        self.model = model.Autoencoder(shape = shape, **self.model_param)

    




    def __init__(self, data_param, model_param, verbose = 1, n_epoch = 3000):
        # basic attributes
        self.data_param = data_param
        self.model_param = model_param
        self.verbose = verbose
        self.n_epoch = n_epoch
        
        # data related attributes
        self.data = None
        self.input_shape = None
        self.metadata = None

        # model related attributes
        self.model = None

        # initializing metrics
        self.metric = None
        self.all_metrics = None
        
        if isinstance(self.data_param, dict):
            self.build_dataset(self.data_param)
        elif isinstance(self.data_param, str):
            self.load_dataset(self.data_param)
        # here we need to capture the shape of the input before building the model.
        self.build_model(shape = self.input_shape, model_param = self.model_param)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, amsgrad=False)
        self.data_set, self.dataloader = helpers.format_dataset(self.data, self.metadata)

        self.monitor = monitoring.Monitor(self.model, self.dataloader, label = self.metadata["subtypes"], verbose= verbose - 1)
        self.callbacks = self.monitor.callbacks
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr= 1e-5)
        
        # useful for VQ-VAE
        self.data_variance = np.var(self.data)

    
        
    def run(self, log = True):
        if self.verbose:
            print("Running the following configuration:")
            print(self.data_param)
            print(self.model_param)

    
        self.model.to(DEVICE)
        for epoch in tqdm(range(self.n_epoch)):
            running_loss = 0.0
            count = 0
            
            # Training loop
            for _, inputs in enumerate(self.dataloader):

                self.optimizer.zero_grad()
                inputs = inputs.to(DEVICE)

                if self.model.variational == "VQ-VAE":
                    quantized_merge = torch.empty(0,1,64).to(DEVICE)

                # Compute the VAE loss or standard loss
                if self.model.variational == "VAE":
                    outputs, mu, log_var = self.model(inputs)
                    reconstruction_loss = F.mse_loss(outputs, inputs)
                    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    loss = reconstruction_loss + (1 * kld)
                elif self.model.variational == "VQ-VAE":
                    vq_loss, data_recon, perplexity, encodings, quantized = self.model(inputs)
                    recon_error = F.mse_loss(data_recon, inputs) / self.data_variance
                    loss = recon_error + vq_loss
                else:
                    outputs = self.model(inputs)
                    loss = F.mse_loss(outputs, inputs)
                
                loss.backward()
                self.optimizer.step()
                count += 1
                running_loss += loss.item()
            
            # Calculate and store training loss for this epoch
            train_loss = running_loss / count
            self.monitor.append_loss(train_loss)
            self.callbacks(epoch)
        
        self.monitor.compute_metrics()
        self.metric = self.monitor.metrics[-1]["ari"]

        if self.verbose:
            visualisation.post_training_viz(self.data, self.dataloader, self.model, DEVICE, self.monitor.train_res_recon_error, labels = self.metadata["subtypes"])
            #visualisation.post_training_animation(self.monitor, self.metadata)
            
            print(self.monitor.metrics[-1])
        
        if log:
            if type(self.data_param) == dict:
                record = {**self.data_param, **self.model_param, **self.monitor.metrics}
            elif type(self.data_param) == str:
                record = {"data" : self.data_param, **self.model_param, **self.monitor.metrics[-1]}
            log_experiment(record)


