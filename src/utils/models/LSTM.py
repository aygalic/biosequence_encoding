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
        self.regularizer = tf.keras.regularizers.L1L2(l1 = 0.05 , l2 = 0.05)  # Adjust the regularization strength as needed

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(shape)),
            layers.LSTM(64, activation='linear', return_sequences=True, kernel_regularizer = self.regularizer, activity_regularizer = self.regularizer),
            layers.LeakyReLU(alpha=0.05),
            layers.Dropout(0.5),

            layers.LSTM(latent_dim, activation='linear', activity_regularizer = self.regularizer),
            layers.LeakyReLU(alpha=0.05),

        ])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        
        self.decoder = tf.keras.Sequential([
            layers.RepeatVector(shape[0]),
            layers.LSTM(64, activation='linear', return_sequences=True),
            layers.LeakyReLU(alpha=0.05),

            layers.LSTM(64, activation='linear', return_sequences=True),
            layers.LeakyReLU(alpha=0.05),

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