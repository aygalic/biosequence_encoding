# import all the things 

import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy


from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras


# model
class CNN_Autoencoder(Model):
    def __init__(self, shape, latent_dim = 64):
        super(CNN_Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Reshape((shape, 1)),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.UnitNormalization(), # to avoid overloading float32
            layers.Conv1D(filters=16, kernel_size=5, padding = "same", activation='selu', kernel_regularizer = tf.keras.regularizers.L2(0.1), bias_regularizer = tf.keras.regularizers.L2(0.1)),
            layers.MaxPooling1D(2),
            layers.Dropout(0.5),
            layers.Conv1D(filters=32, kernel_size=3, padding = "same", activation='selu', kernel_regularizer = tf.keras.regularizers.L2(0.1), bias_regularizer = tf.keras.regularizers.L2(0.1)),
            layers.AveragePooling1D(2),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='softplus'),
        ])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        
        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dim, activation='softplus'),
            layers.Reshape((latent_dim, 1)),
            layers.Conv1D(filters=32, kernel_size=3, padding = "same", activation='selu'),
            layers.UpSampling1D(2),
            layers.Conv1D(filters=16, kernel_size=5, padding = "same", activation='selu'),
            layers.UpSampling1D(2),
            layers.Flatten(),
            layers.Dense(1 * shape, activation='softplus'), # softplus so we can have value in the expected range
        ])
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker
        ]

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            total_loss = losses.mean_squared_error(reconstruction, data) 
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result()
        }


def generate_model(shape, latent_dim = 64):
    model = CNN_Autoencoder(shape, latent_dim)
    return model