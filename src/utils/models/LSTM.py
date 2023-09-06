# import all the things 

import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy


from tensorflow.keras import layers, losses, initializers
from tensorflow.keras.models import Model
from tensorflow import keras


# model
class LSTM_Autoencoder(Model):
    def __init__(self, shape, latent_dim = 64):
        super(LSTM_Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.regularizer = tf.keras.regularizers.L1L2(l1 = 0.05 , l2 = 0.05)  # Adjust the regularization strength as needed

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(shape)),
            layers.LSTM(1024, activation='tanh', return_sequences=True, kernel_regularizer = self.regularizer, activity_regularizer = self.regularizer),
            layers.LeakyReLU(alpha=0.05),
            layers.Dropout(0.5),

            layers.LSTM(512, activation='tanh', return_sequences=True, kernel_regularizer = self.regularizer, activity_regularizer = self.regularizer),
            layers.LeakyReLU(alpha=0.05),
            layers.Dropout(0.5),

            layers.LSTM(256, activation='tanh', return_sequences=True, kernel_regularizer = self.regularizer, activity_regularizer = self.regularizer),
            layers.LeakyReLU(alpha=0.05),
            layers.Dropout(0.5),

            layers.LSTM(latent_dim, activation='tanh', activity_regularizer = self.regularizer),
            layers.LeakyReLU(alpha=0.05),

        ])
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        
        self.decoder = tf.keras.Sequential([
            layers.RepeatVector(shape[0]),
            layers.LSTM(256, activation='tanh', return_sequences=True),
            layers.LeakyReLU(alpha=0.05),

            layers.LSTM(512, activation='tanh', return_sequences=True),
            layers.LeakyReLU(alpha=0.05),

            layers.LSTM(1024, activation='tanh', return_sequences=True),
            layers.LeakyReLU(alpha=0.05),

            layers.TimeDistributed(tf.keras.layers.Dense(shape[1], activation='relu'))  # Use 'relu' activation


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


class biDir_LSTM_Autoencoder(Model):
    def __init__(self, shape, latent_dim = 64):
        super(biDir_LSTM_Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.regularizer = tf.keras.regularizers.L1L2(l1 = 0.05 , l2 = 0.05)  # Adjust the regularization strength as needed

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(shape)),
            layers.Bidirectional(layers.LSTM(128, activation='tanh', return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64, activation='tanh', return_sequences=True)),


            layers.Bidirectional(layers.LSTM(latent_dim, activation='tanh', activity_regularizer=self.regularizer)),
            #layers.LeakyReLU(alpha=0.05),
        ])

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        self.decoder = tf.keras.Sequential([
            layers.RepeatVector(shape[0]),
            layers.Bidirectional(layers.LSTM(64, activation='tanh', return_sequences=True)),
            layers.Bidirectional(layers.LSTM(128, activation='tanh', return_sequences=True)),



            layers.TimeDistributed(tf.keras.layers.Dense(shape[1], activation='linear')),
            layers.LeakyReLU(alpha=0.05)
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


def generate_model(shape, latent_dim = 64, bidirrectional = False):
    if(bidirrectional == True):
        model = biDir_LSTM_Autoencoder(shape, latent_dim)

    else:
        model = LSTM_Autoencoder(shape, latent_dim)
    return model