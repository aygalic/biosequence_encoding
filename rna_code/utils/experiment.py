from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from rna_code.data.data_module import DataModule
from rna_code.utils.monitor_callback import MonitorCallback

from .. import DEVICE, LOGFILE
from ..models import autoencoder
from . import visualisation


class Experiment():
    def __init__(
            self,
            data_param : dict,
            model_param : dict
            ) -> None:
        self.data_param = data_param
        self.model_param = model_param
        # FIXME n_epoch is not actually a model param, this was made to simplify code
        self.n_epoch = self.model_param.pop("n_epoch", 1000)

        self.data_module = DataModule(data_param)
        self.data_module.setup(stage=None)
        self.input_shape = self.data_module.feature_num
        self.model = autoencoder.Autoencoder(shape = self.input_shape, **self.model_param)

    def run(self):
        labels = self.data_module.full_meta_data["subtypes"]
        labels = [label if label is not np.nan else "None" for label in labels]
        unique_labels = {l:i for (i,l) in enumerate(np.unique(labels))}
        processed_labels = [unique_labels[l] for l in labels]

        monitor_callback = MonitorCallback(self.data_module.full_data_loader(), processed_labels, n_clusters=10)

        trainer = pl.Trainer(max_epochs=10, callbacks=[monitor_callback])
        trainer.fit(self.model, self.data_module)
        visualisation.post_training_viz(
            data = self.data_module.data_array,
            dataloader = self.data_module.full_data_loader(),
            model =  self.model,
            DEVICE = DEVICE,
            loss_hist = monitor_callback.loss_values,
            labels = labels
            )
        
        print(monitor_callback.metrics[-1])

        if isinstance(self.data_param, dict):
            record = {**self.data_param, **self.model_param, **monitor_callback.metrics[-1]}
        elif isinstance(self.data_param, str)  or isinstance(self.data_param, Path):
            record = {"data" : self.data_param, **self.model_param, **monitor_callback.metrics[-1]}

        Experiment._log_experiment(record)

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
