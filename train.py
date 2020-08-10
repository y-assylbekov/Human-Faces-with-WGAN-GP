"""
author: Yernat M. Assylbekov
email: yernat.assylbekov@gmail.com
date: 08/10/2020
"""


import numpy as np
import tensorflow as tf
from model import Generator, Critic, loss_generator, loss_critic, create_generator, create_critic
from utils import read_preprocess_images, print_save_images
from IPython import display
import os
import matplotlib.pyplot as plt


def generate_and_print(model, input, epoch):
    """
    generates, prints and saves 25 images for a given input noise of shape [25, 128] at a current epoch.
    """
    prediction = model(input, training=False)
    fig = plt.figure(figsize=(10, 10))

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(prediction[i])
        plt.axis('off')

    fig.suptitle('Epoch {}'.format(epoch), fontsize=16, y=0.92)
    plt.savefig('image_at_epoch_{:03d}.png'.format(epoch))
    plt.show()

def train_WGAN_GP(train_set, batch_size, epochs):
    """
    trains the globally defined generator and critic.
    """
    # training set size
    train_size = train_set.shape[0]

    # number of mini batches
    m = train_size // batch_size

    # partition the training set into mini batches
    train_batches = np.split(train_set, [k * batch_size for k in range(1, m)])

    # generator and critic and their optimizers are given globally
    global generator, generator_optimizer
    global critic, critic_optimizer

    # lists to record costs of the generator and critic at every epoch
    generator_costs = list()
    critic_costs = list()

    for epoch in range(1, epochs + 1):
        # initiate costs of the generator and critic at the current epoch to zeros
        generator_cost = 0
        critic_cost = 0

        for batch in train_batches:
            # sample random noise for the current mini batch
            noise = tf.random.normal([batch_size, 128])

            # watch trainable variables for the loss functions of the generator and critic
            with tf.GradientTape() as generator_tape, tf.GradientTape() as critic_tape:
                fake_images = generator(noise, training=True)

                # the critic computes logits of images for being real
                logit_fake = critic(fake_images, training=True)
                logit_real = critic(batch, training=True)

                # loss functions for the generator and critic
                generator_loss = loss_generator(logit_fake)
                critic_loss = loss_critic(logit_real, logit_fake, batch, fake_images, partial(critic, training=True))

            # compute gradients and perform one step gradient descend
            critic_Grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_Grads, critic.trainable_variables))
            generator_Grads = generator_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(generator_Grads, generator.trainable_variables))

            # record costs of the generator and critic at every epoch
            fake_images = generator(noise, training=False)
            logit_fake = critic(fake_images, training=False)
            logit_real = critic(batch, training=False)
            generator_cost += loss_generator(logit_fake).numpy() / m
            critic_cost += loss_critic(logit_real, logit_fake, batch, fake_images, partial(critic, training=False)).numpy() / m

        # print and save 25 randomly generated images at every epoch
        generator_costs.append(generator_cost)
        critic_costs.append(critic_cost)

        display.clear_output(wait=True)
        print('Epoch: {}'.format(epoch))
        print('Generator loss: {}'.format(generator_loss))
        print('Critic loss: {}'.format(critic_loss))
        noise = tf.random.normal([25, 128])
        generate_and_print(generator, noise, epoch)

    # save the weights of the generator and critic at after training
    generator.save_weights('./generator_weights/')
    critic.save_weights('./critic_weights/')

    # plot the learning curves of the generator and critic
    plt.plot(np.squeeze(generator_costs), '-b', label='Generator')
    plt.plot(np.squeeze(critic_costs), '-r', label='Critic')
    plt.legend(loc='upper left')
    plt.title('Learning Curve')
    plt.savefig('learning_curve.png')
    plt.show()


# download and preprocess images
path = 'img_align_celeba/img_align_celeba/*.jpg'
train_set = read_images(path, n=131072)

# let us look at few images from the dataset
print_save_images(train_set)

# setup learning rate, beta_1 and beta_2
learning_rate, beta_1, beta_2 = 0.0002, 0.5, 0.9

# create instances of generator and critic and their optimizers
generator, generator_optimizer = create_generator(learning_rate, beta_1, beta_2)
critic, critic_optimizer = create_critic(learning_rate, beta_1, beta_2)

# train generator and critic
train_WGAN_GP(train_set, batch_size=64, epochs=100)
