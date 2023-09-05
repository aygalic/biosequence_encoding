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



class VAE(keras.Model):
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

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                    self.loss(data, reconstruction), axis=(0)
                )
            #reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction) # which one is right ???
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


    # default encoder
    encoder_inputs = keras.Input(shape=(shape,))
    x = layers.UnitNormalization()(encoder_inputs) # to avoid overflowing float32
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")




    # default decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64)(latent_inputs)
    x = layers.LeakyReLU(alpha=0.05)(x)

    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)

    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.05)(x)

    decoder_outputs = layers.Dense(shape)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder)
    return vae