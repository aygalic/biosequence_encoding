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
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='selu', padding='same', input_shape=shape),
            layers.MaxPooling1D(2, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='selu', padding='same', input_shape=shape),
            tf.keras.layers.Flatten(),
            #tf.keras.layers.Dense(latent_dim, activation = "tanh"), # this dooms the learning process
            tf.keras.layers.Dense(latent_dim),
        ])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        
        self.decoder = tf.keras.Sequential([
            layers.Reshape((2, int(latent_dim/2))),
            layers.Conv1D(filters=32, kernel_size=3, padding = "same", activation='selu'),
            layers.UpSampling1D(2),            
            layers.Conv1D(filters=16, kernel_size=5, padding = "same", activation='selu'),
            layers.UpSampling1D(2),
            #layers.UpSampling1D(1),  # Use 1 for upsampling factor
            layers.Cropping1D(cropping=(1, 2)),            
            layers.Conv1D(filters=shape[1], kernel_size=5, padding="same", activation='sigmoid'),  # Change kernel size here


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
    model._name = "CNN_Autoencoder"
    model._is_variational = False
    return model