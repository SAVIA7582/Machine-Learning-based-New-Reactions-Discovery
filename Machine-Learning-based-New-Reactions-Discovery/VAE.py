from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.regularizers import l1

from pipe import Pipe

import numpy as np


def to_ndarray(x: list):
    return np.array(x, dtype=np.float32)


def create_vae(input_dim, latent_dim, output_dim):
    """
    create vae model with encoder and generator
    :param input_dim: dimension of input datas
    :param latent_dim: dimension of latent layer
    :param output_dim: dimension of output datas
    :return: encoders and decoders
    """
    # encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(256, activation='relu', activity_regularizer=l1(10e-5))(input_layer)
    encoded = Dense(128, activation='relu', activity_regularizer=l1(10e-5))(encoded)

    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # decoder
    decoder_h1 = Dense(128, activation='relu', activity_regularizer=l1(10e-5))
    decoder_h2 = Dense(256, activation='relu', activity_regularizer=l1(10e-5))
    decoder_out = Dense(output_dim, activation='sigmoid')

    decoded = decoder_h1(z)
    decoded = decoder_h2(decoded)
    decoded = decoder_out(decoded)

    # Building models
    vae = Model(input_layer, decoded)
    encoder = Model(input_layer, z_mean)

    decoder_input = Input(shape=(latent_dim,))
    _decoded = decoder_h1(decoder_input)
    _decoded = decoder_h2(_decoded)
    _decoded = decoder_out(_decoded)
    generator = Model(decoder_input, _decoded)

    # Define loss function
    def vae_loss(x, decoded_x):
        xent_loss = objectives.binary_crossentropy(x, decoded_x)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='adam', loss=vae_loss)

    return vae, encoder, generator


def create_autoencoder(input_dim):

    """
    generate a shallow autoencoder
    :param input_dim: dimension of input datas
    :return: two encoders
    """

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(input_dim, activation='relu', activity_regularizer=l1(10e-5))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder


def predictor(input_dim, latent_dim, output_dim=256):

    """
    build specific prediction function of encoders and decoders wrapped with pipes
    :param input_dim: dimension of input datas
    :param latent_dim: dimension of latent layers
    :param output_dim: dimension of output datas
    :return: 3 prediction functions wrapped with pipe
    """

    _, encoder, generator = create_vae(input_dim, latent_dim, output_dim)
    _, autoencoder_encoder = create_autoencoder(input_dim)

    @Pipe
    def autoencoder_predict(x: list):
        return autoencoder_encoder.predict(to_ndarray(x).reshape(1, -1)).tolist()

    @Pipe
    def vae_predict(x: list):
        return encoder.predict(to_ndarray(x)).tolist()

    @Pipe
    def generator_predict(x):
        return generator.predict(to_ndarray(x)).tolist()[0]

    return autoencoder_predict, vae_predict, generator_predict
