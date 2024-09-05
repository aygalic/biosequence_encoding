"""
Experiment Module for Model Training and Evaluation

This module provides functionalities to set up, execute, and log experiments.

Classes:
    Experiment: A class for managing the overall process of setting up, 
    running, and logging model training with specific data and model configurations.

Functions:
    log_experiment: Function to log the results of an experiment to a CSV file.

Usage:
    The module is used to create an instance of an Experiment class, which can 
    then be configured with specific data and model parameters. The experiment 
    can be executed, and the results are logged for analysis.

Example:
    To use this module, import it in your script, initialize an Experiment 
    object with the desired parameters, and call its `run` method to start 
    the experiment. Optionally, you can log the results using `log_experiment`.

    >>> experiment = Experiment(data_param, model_param)
    >>> experiment.run(log=True)

Note:
    This module assumes the availability of certain external libraries and 
    modules, such as PyTorch for model building and training, and pandas 
    for data handling. Ensure these dependencies are met before using this module.
"""



import sys
import pickle

sys.path.append('..')
#from utils import data_handler
from ..data import data_handler
from . import visualisation, helpers, monitoring
from ..models import autoencoder

from .. import DEVICE, LOGFILE

# data manipulation
import pandas as pd
import numpy as np


# pytorch specific
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


def log_experiment(record, csv_path=LOGFILE):
    """
    Logs the experiment results to a CSV file.

    Args:
        record (dict): A dictionary containing the experiment's results and parameters.
        csv_path (str): Filepath to the CSV file where the log will be stored.

    Note:
        If the CSV file does not exist, it creates a new file. If it exists, the new record is appended.
    """
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
    """
    A class for setting up and running experiments with specific data and model configurations.

    Attributes:
        data_param (dict or str): Parameters for the dataset or filepath to a saved dataset.
        model_param (dict): Parameters for model configuration.
        verbose (int): Verbosity level for output messages.
        data (array): Loaded dataset.
        input_shape (int): Shape of the input data.
        metadata (dict): Metadata associated with the dataset.
        model (torch.nn.Module): The model for the experiment.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        monitor (Monitor): Monitoring object for tracking training progress.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        data_variance (float): Variance of the dataset, used in VQ-VAE.

    Methods:
        build_dataset: Builds the dataset from specified parameters.
        load_dataset: Loads the dataset from a pickle file.
        build_model: Initializes the model based on input shape and model parameters.
        run: Executes the training process and logs results if specified.
    """

    def __init__(self, data_param, model_param, verbose = 1):
        # basic attributes
        self.data_param = data_param
        self.model_param = model_param
        # n_epoch is not actually a model param, but this was made for simpler code, default value : 1000
        self.n_epoch = self.model_param.pop("n_epoch", 1000)

        self.verbose = verbose
        
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
        self.data_set, self.dataloader = helpers.format_dataset(self.data)

        self.monitor = monitoring.Monitor(self.model, self.dataloader, label = self.metadata["subtypes"], verbose= verbose - 1)
        self.callbacks = self.monitor.callbacks
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr= 1e-5)
        
        # useful for VQ-VAE
        self.data_variance = np.var(self.data)

        
    def build_dataset(self, data_param):
        data_param["verbose"] = self.verbose - 1
        self.data, self.metadata = data_handler.generate_dataset(**data_param)
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

        self.model = autoencoder.Autoencoder(shape = shape, **self.model_param)

    
    def run(self, log = True):
        """
        Runs the experiment with the specified model and dataset.

        This method manages the training loop, loss calculation, and calls to monitor callbacks. 
        It also handles visualization and logging of results based on verbosity and logging settings.

        Args:
            log (bool): Whether to log the experiment results to a file.

        Note:
            The method assumes that the model, data, and other necessary components are already set up.
        """
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
                record = {**self.data_param, **self.model_param, **self.monitor.metrics[-1]}
            elif type(self.data_param) == str:
                record = {"data" : self.data_param, **self.model_param, **self.monitor.metrics[-1]}
            log_experiment(record)


