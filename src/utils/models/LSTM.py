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
class Autoencoder(Model):
    def __init__(self, shape, latent_dim = 64):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(shape)),
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.LSTM(latent_dim, activation='relu'),
        ])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        
        self.decoder = tf.keras.Sequential([
            layers.RepeatVector(shape[0]),
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.TimeDistributed(tf.keras.layers.Dense(shape[1]))
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
    model = Autoencoder(shape, latent_dim)
    return model