"""Experiment module, in charge of handling data loading, model training and monitoring
according to provided parameters.
"""
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import numpy as np

from rna_code.data.data_module import DataModule
from rna_code.utils.monitor_callback import MonitorCallback

from .. import DEVICE, LOGFILE

from ..models.model_builder import ModelBuilder
from . import visualization
from .experiment import Experiment

class TransfertLearningExperiment(Experiment):
    """Experiment class handling data, model training and monitoring.

    Parameters
    ----------
    data_param : dict
        Data related parameters.
    model_param : dict
        Model related parameters.
    """
    def __init__(
            self,
            data_param : dict,
            model_param : dict
            ) -> None:

        self.data_param = data_param
        self.model_param = model_param

        self.n_epoch = self.model_param.pop("n_epoch", 10)
        self.data_module = DataModule(data_param)
        self.data_module.setup(stage=None)
        self.input_shape = self.data_module.feature_num

        self.model_builder = ModelBuilder(self.input_shape, self.model_param)
        self.model : pl.LightningModule


    def run(self) -> None:
        """Run experiment.

        Training, visualization and logging.
        """
        self.model = self.model_builder.generate_model()

        monitoring_interval = np.unique([int(x) for x in np.logspace(1, 3, num=50)])
        monitor_callback = MonitorCallback(
             dataloader=self.data_module.full_data_loader(),
             labels=self.data_module.full_meta_data["subtypes"],
             n_clusters=5,
             compute_on='batch',
             evaluation_intervals = monitoring_interval,
             verbose=0
        )
        trainer = pl.Trainer(max_epochs=self.n_epoch, callbacks=[monitor_callback])
        trainer.fit(self.model, self.data_module)
        visualization.post_training_viz(
            data = self.data_module.data_array,
            dataloader = self.data_module.full_data_loader(),
            model =  self.model,
            DEVICE = DEVICE,
            loss_hist = monitor_callback.loss_values,
            labels = self.data_module.full_meta_data["subtypes"]
            )
        print(monitor_callback.metrics[-1])
        record = {**self.data_param, **self.model_param, **monitor_callback.metrics[-1]}
        Experiment._log_experiment(record)
        visualization.post_training_animation(
            monitor = monitor_callback,
            metadata = self.data_module.full_meta_data)


