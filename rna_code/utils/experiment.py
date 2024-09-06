from pathlib import Path
import sys
import pickle
import pytorch_lightning as pl

#sys.path.append('..')
from ..data import data_handler
from . import visualisation, helpers
from ..models import autoencoder

from .. import DEVICE, LOGFILE

import pandas as pd
import numpy as np


from rna_code.utils.monitor_callback import MonitorCallback
from rna_code.data.data_module import DataModule


class Experiment():


    def __init__(self, data_param, model_param):
        self.data_param = data_param
        self.model_param = model_param
        # n_epoch is not actually a model param, but this was made for simpler code, default value : 1000
        self.n_epoch = self.model_param.pop("n_epoch", 1000)
        
        if isinstance(self.data_param, dict):
            self._build_dataset(self.data_param)
        elif isinstance(self.data_param, str) or isinstance(self.data_param, Path):
            self._load_dataset(self.data_param)

        self.input_shape = len(self.metadata["feature_names"])
        print("input shape :", self.input_shape)

        # here we need to capture the shape of the input before building the model.
        self._build_model(shape = self.input_shape, model_param = self.model_param)
        self.data_set, self.dataloader = helpers.format_dataset(self.data)


    def run(self):
        data_module = DataModule()
        data_module.setup(stage=None)
        labels = data_module.train_meta_data["subtypes"]
        labels = [label if label is not np.nan else "None" for label in labels]
        unique_labels = {l:i for (i,l) in enumerate(np.unique(labels))}
        processed_labels = [unique_labels[l] for l in labels]

        monitor_callback = MonitorCallback(data_module.train_dataloader(), processed_labels, n_clusters=10)

        trainer = pl.Trainer(max_epochs=10, callbacks=[monitor_callback])
        trainer.fit(self.model, data_module)
        visualisation.post_training_viz(
            data = self.data,
            dataloader = self.dataloader,
            model =  self.model,
            DEVICE = DEVICE,
            loss_hist = monitor_callback.loss_values,
            labels = self.metadata["subtypes"]
            )
        
        print(monitor_callback.metrics[-1])
        

        if isinstance(self.data_param, dict):
            record = {**self.data_param, **self.model_param, **monitor_callback.metrics[-1]}
        elif isinstance(self.data_param, str)  or isinstance(self.data_param, Path):
            record = {"data" : self.data_param, **self.model_param, **monitor_callback.metrics[-1]}

        Experiment._log_experiment(record)


    def _build_dataset(self, data_param):
        self.data, self.metadata = data_handler.generate_dataset(**data_param)

    def _load_dataset(self, data_param):
        data_path =  data_param / 'data_array.npy'
        metadata_path = data_param / 'meta_data.json'
        self.data = np.load(data_path)
        with metadata_path.open('rb') as f:
            self.metadata = pickle.load(f)

    def _build_model(self, shape, model_param):
        if model_param.get("transformer", False) == True:
            num_heads_candidate = helpers.find_primes(self.input_shape)
            if(len(num_heads_candidate) > 1):
                self.model_param["num_heads"] = num_heads_candidate[-1]
            else:
                self.model_param["num_heads"] = num_heads_candidate[-2]

        self.model = autoencoder.Autoencoder(shape = shape, **self.model_param)

    @staticmethod
    def _log_experiment(record : dict, csv_path : Path = LOGFILE):
        """Logs the experiment results to a csv file.

        Parameters
        ----------
        record : dict
            Data to append to logfile
        csv_path : Path, optional
            Path of the file, by default LOGFILE
        """
        new_df = pd.DataFrame([record])
        try:
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        except FileNotFoundError:
            combined_df = new_df
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(csv_path, index=False)