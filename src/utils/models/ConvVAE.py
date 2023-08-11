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
class Sampling(layers.Layer):
    #Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# model
class ConvVAE(Model):
    def __init__(self, encoder, decoder, loss = keras.losses.mean_squared_error, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #reconstruction_loss = tf.reduce_mean(
            #        self.loss(data, reconstruction), axis=(0)
            #    )
            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction) # which one is right ???
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def generate_model(shape, latent_dim = 64):


    #  encoder
    encoder_inputs = keras.Input(shape=(shape))
    x = layers.Conv1D(filters=64, kernel_size=5, activation='linear', padding='same', input_shape=shape)(encoder_inputs)
    x = layers.LeakyReLU(alpha=0.05)(x)

    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='linear', padding='same', input_shape=shape)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)


    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")




    #  decoder
                


    latent_inputs = keras.Input(shape=(latent_dim))

    x = layers.Reshape((2, int(latent_dim/2)))(latent_inputs)
    x = layers.UpSampling1D(2)(x)           
    x = layers.Conv1D(filters=16, kernel_size=5, padding = "same", activation='selu')(x)
    x = layers.UpSampling1D(2)(x)

    x = layers.Cropping1D(cropping=(1, 2))(x)          
    decoder_outputs = layers.Conv1D(filters=shape[1], kernel_size=5, padding="same", activation='selu')(x)





    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = ConvVAE(encoder, decoder)
    return vae










    model = CNN_Autoencoder(shape, latent_dim)
    return model