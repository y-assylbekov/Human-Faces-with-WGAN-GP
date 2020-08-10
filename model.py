"""
author: Yernat M. Assylbekov
email: yernat.assylbekov@gmail.com
date: 08/10/2020
"""


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Flatten, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

def loss_generator(logit_fake):
    """
    loss function for generator.
    """
    return - tf.math.reduce_mean(logit_fake)

def loss_critic(logit_real, logit_fake, real, fake, critic):
    """
    loss function with gradient penalty for critic.
    """
    # interpolate between real and fake (stochastically via uniform distribution)
    alpha = tf.random.uniform([real.shape[0], 1, 1, 1], 0., 1.)
    interp = alpha * real + (1 - alpha) * fake

    # compute the gradient of critic w.r.t. interp
    with tf.GradientTape() as t:
        t.watch(interp)
        logit_inter = critic(interp)
    grad = t.gradient(logit_inter, [interp])[0]

    # compute the norm of the gradient
    slopes = tf.math.sqrt(tf.math.reduce_sum(grad ** 2, axis=[1, 2, 3]))

    return - tf.math.reduce_mean(logit_real) + tf.math.reduce_mean(logit_fake) + 10. * tf.math.reduce_mean((slopes - 1.) ** 2)

def Generator():
    """
    model for generator.
    """

    # set up input
    X = Input(shape=128)

    # project to 4x4
    Y = Dense(units=4*4*256)(X)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)
    Y = Reshape(target_shape=(4, 4, 256))(Y)

    # map to 8x8
    Y = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same')(Y)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)

    # map to 16x16
    Y = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same')(Y)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)

    # map to 32x32
    Y = Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same')(Y)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)

    # map to 64x64
    Y = Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='sigmoid')(Y)

    model = Model(inputs=X, outputs=Y)

    return model


def Critic():
    """
    model for critic.
    """

    X = Input(shape=(64, 64, 3))

    # map to 32x32
    Y = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(X)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)
    Y = Dropout(0.5)(Y)

    # map to 16x16
    Y = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(Y)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)
    Y = Dropout(0.5)(Y)

    # map to 8x8
    Y = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(Y)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)
    Y = Dropout(0.5)(Y)

    # map to 4x4
    Y = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(Y)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)
    Y = Dropout(0.5)(Y)

    # map to 2x2
    Y = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(Y)
    Y = BatchNormalization()(Y)
    Y = LeakyReLU()(Y)
    Y = Dropout(0.5)(Y)

    Y = Flatten()(Y)
    Y = Dense(units=1)(Y)

    model = Model(inputs=X, outputs=Y)

    return model


def create_generator(learning_rate, beta_1, beta_2):
    """
    creates generator and its optimizer (Adam).
    """
    generator = Generator()
    generator_optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2)

    return generator, generator_optimizer

def create_critic(learning_rate, beta_1, beta_2):
    """
    creates critic and its optimizer (Adam).
    """
    critic = Critic()
    critic_optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2)

    return critic, critic_optimizer
