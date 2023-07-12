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


# datasets

# default path of the folder containing the salmon files
absolute_path = '/Users/aygalic/OneDrive/polimi/Thesis/data/quant/'  





# from filename to tensor
# here we open a single file passed as "filename" we return a tensor of TPM values.
def load_patient_data(filename):
  #specify read types for our data
  read_types = [float()]
  # get a first sample to base everything of
  text = pathlib.Path(absolute_path + filename).read_text()
  lines = text.split('\n')[1:-1]
  # the 3rd column correspond to TPM values.
  features = tf.io.decode_csv(lines, record_defaults=read_types, field_delim = "\t", select_cols=[3])
  data = tf.convert_to_tensor(features)[0]
  return data


# building the actual dataset
def generate_dataset(path = absolute_path, feature_selection_threshold = None, batch_size = 64, subsample = None):
    # getting entries ready
    # each couple of entries correspond to one patient, we are only interested in the "transcript" files
    entries = os.listdir(absolute_path)
    entries_transcripts = [e for e in entries if "transcripts" in e ]
    if(subsample is not None):
        entries_transcripts = entries_transcripts[1:40] # for testing purpose 
    # load the dataset into a list using the first pipeline
    data = [load_patient_data(e) for e in entries_transcripts]

    # remove artifacts by keeping samples of correct length
    samples_to_keep = [1 if s.shape == (95309) else 0 for s in data]
    train_ds = [sample for (sample, test) in  zip(data, samples_to_keep) if test]

    # if feature selection is applied
    if(feature_selection_threshold is not None):
        data_array = np.array(train_ds)
        MAD = scipy.stats.median_abs_deviation(data_array)
        gene_selected = [True if val > feature_selection_threshold else False for val in MAD]
        print("number of genes selected : ",sum(gene_selected))
        train_ds = data_array[:,gene_selected]

    x_train = tf.data.Dataset.from_tensor_slices(train_ds)
    dataset = x_train.batch(batch_size)

    return dataset



# models

class Sampling(layers.Layer):
    #Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class Encoder():
    def __init__(self, input_shape, latent_dim = 64, **kwargs):
        self.input_shape = input_shape
        self.encoder_inputs = keras.Input(shape=input_shape)
        x = layers.Flatten()(self.encoder_inputs)
        x = layers.UnitNormalization()(x) # to avoid overloading float32
        x = layers.Dense(256, activation = "ELU")(x)
        x = layers.Dense(128, activation = "ELU")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation = "ELU")(x)
        x = layers.Dense(64, activation = "ELU")(x)
        self.z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        self.z = Sampling()([self.z_mean, self.z_log_var])
        self.encoder = keras.Model(self.encoder_inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")
        #self.encoder.summary()

class Decoder():
    def __init__(self, output_shape, latent_dim = 64, **kwargs):
        self.latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(64, activation="ELU")(self.latent_inputs)
        x = layers.Dense(128, activation="ELU")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="ELU")(x)
        x = layers.Dense(256, activation="ELU")(x)
        self.decoder_outputs = layers.Dense(output_shape, activation="ELU")(x)
        self.decoder = keras.Model(self.latent_inputs, self.decoder_outputs, name="decoder")
        #self.decoder.summary()





class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
                    keras.losses.mean_squared_error(data, reconstruction), axis=(0)
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
    encoder = Encoder(shape, latent_dim).encoder
    decoder = Decoder(shape, latent_dim).decoder
    vae = VAE(encoder, decoder)
    return vae