# Copyright 2022 Softpoint Consultores SL. All Rights Reserved.
#
# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import re
import gc
import logging
import warnings
from itertools import compress
from numpy.random import randn
import numpy as np
import tensorflow as tf
from keras.initializers import RandomNormal, RandomUniform
from keras.layers import Dense, Dropout, LeakyReLU, \
    BatchNormalization, Conv2D, Flatten
from keras import backend as K
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tqdm import tqdm


class WGAN(object):
    """ 
    Wasserstein GAN leverages the Wasserstein distance to 
    produce a value function that has better theoretical 
    properties than the value function used in the original 
    GAN paper. WGAN requires that the discriminator 
    lie within the space of 1-Lipschitz functions
    (any function with a bounded first derivative). 
    It means that, for a given function f(x), there exist 
    constants m and M such that f′(x) always satisfies 
    m ≤ f'(x) ≤ M.
    References:
    Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein 
    generative adversarial networks. In  International 
    conference on machine learning (pp. 214-223). PMLR.
    """

    def __init__(self, number_of_features, learning_rate, dropout, show_tqdm: bool = True):
        self.number_of_features = number_of_features
        self.G_model = None
        self.noise_dim = None
        self.D_model = None
        self.learning_rate = learning_rate
        self.gan_model = None
        self.dropout = dropout
        self.number_of_features = number_of_features
        self.show_tqdm = show_tqdm
        self.build_generator()
        self.build_discriminator()
        self.build_wgan()

    def build_generator(self):
        """ We create Generator which uses MLP with
        simple dense layers activated by tanh.
        return: generator model, G. It is used to 
        generate new plausible examples from 
        the problem domain.
        D_z is the dimensionality of noise prior
        or latent space, z, with prior density pz(z).
        We will use it for noise injection as an effective 
        strategy to avoid overfitting and enhancing generalization."""

        D_z = int((self.number_of_features * 3) / 4)
        initializer = RandomNormal(mean=0.0, stddev=0.02)
        self.noise_dim = (D_z,)
        self.G_model = Sequential()
        self.G_model.add(
            Dense(
                256,
                input_shape=self.noise_dim,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )
        )
        self.G_model.add(LeakyReLU(alpha=0.7))
        self.G_model.add(BatchNormalization(momentum=0.9))
        self.G_model.add(Dense(512))
        self.G_model.add(LeakyReLU(alpha=0.7))
        self.G_model.add(BatchNormalization(momentum=0.9))
        self.G_model.add(Dense(1024))
        self.G_model.add(LeakyReLU(alpha=0.7))
        self.G_model.add(BatchNormalization(momentum=0.9))

        # Compile it
        self.G_model.add(
            Dense(
                self.number_of_features,
                activation='tanh'
            )
        )

    def wasserstein_loss(self, y_true, y_pred):
        """
          Calculates the Wasserstein loss for a sample batch.
          The Wasserstein loss function is straightforward to compute. 
          The discriminator in a standard GAN has a sigmoid output, 
          which represents the probability that samples are real or generated.
          Wasserstein GAN output is linear, with no activation function. 
          And, rather than being limited to [0, 1], the discriminator 
          seeks to maximize the distance between its output for 
          real and generated samples. Because of the nature of 
          this loss, it can (and frequently will) be less than 0.
        """
        w_loss = K.mean(y_true * y_pred)
        return w_loss

    def build_discriminator(self):
        """
        We now create the Discriminator which is also MLP. 
        D will take the input from real data  
        and also the data generated from G.  G is used to 
        classify examples as real (from the domain) 
        or fake (generated).
        """
        initializer = RandomNormal(mean=0.0, stddev=0.02)
        self.D_model = Sequential()
        self.D_model.add(Dense(
            512,
            input_shape=(self.number_of_features,),
            kernel_initializer=initializer,
            bias_initializer='zeros'
        )
        )
        self.D_model.add(LeakyReLU(alpha=0.7))
        self.D_model.add(Dropout(self.dropout))
        self.D_model.add(Dense(256))
        self.D_model.add(LeakyReLU(alpha=0.7))
        self.D_model.add(Dropout(self.dropout))
        self.D_model.add(Dense(256))
        self.D_model.add(LeakyReLU(alpha=0.7))
        self.D_model.add(Dropout(self.dropout))

        # Activation function can't be a sigmoid. We use
        # instead a linear function in 1
        self.D_model.add(Dense(1, activation='linear'))
        optimizer = RMSprop(
            learning_rate=0.001,
            rho=0.9,
            momentum=0,
            epsilon=1e-07,  # makes training more stable
            centered=True,  # makes training more stable
            clipvalue=0.01  # gradients are cliped to -0.01 - 0.01
        )
        self.D_model.compile(
            loss=self.wasserstein_loss,  # wassertein loss
            optimizer=optimizer,
            run_eagerly=True  # debugging
        )

    def build_wgan(self):
        """
        Create the GAN model. The GAN model architecture 
        involves two sub-models: a generator model for 
        generating new examples and a discriminator model 
        for classifying whether generated examples are real, 
        from the domain, or fake, generated by 
        the generator model.
        """
        # We freeze the discriminator's layers since we're
        # only interested in the generator and its learning.
        self.D_model.trainable = False
        self.wgan_model = Sequential()

        # Connect the G and D models to the GAN.
        self.wgan_model.add(self.G_model)
        self.wgan_model.add(self.D_model)

        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=0.4,
            amsgrad=True
        )
        self.wgan_model.compile(
            loss=['binary_crossentropy'],
            optimizer=optimizer
        )

        return self.wgan_model

    def train(self, scaled_data, epochs, batch_size):
        """
        This function trains the G and D outputs
        Args:
            scaled_data (np.array):
                input data transformed
            epochs:
                number of epochs
            batch_size:
                bacth size

        Returns:
            data sample and labels array
        """
        discriminator_loss, generator_loss = [], []
        iterations = int(
            len(scaled_data) / int(batch_size * 0.5)
        )
        if len(scaled_data) % batch_size != 0:
            iterations += 1
        else:
            iterations

        for epoch in range(1, epochs + 1):
            np.random.shuffle(scaled_data)
            bar = tqdm(
                range(iterations),
                ascii=True
            ) if self.show_tqdm else range(iterations)

            for iteration in bar:
                dis_loss,\
                    gen_loss = self.train_models(
                        batch_size=batch_size,
                        index=iteration,
                        scaled_data=scaled_data
                    )
                discriminator_loss.append(dis_loss)
                generator_loss.append(gen_loss)
                if self.show_tqdm:
                    bar.set_description(
                        f"Epoch ({epoch}/{epochs}) | D. loss: {dis_loss:.2f} | G. loss: {gen_loss:.2f} |")
        return generator_loss[-1], discriminator_loss[-1]

    def train_models(
        self,
        batch_size,
        index,
        scaled_data
    ):
        """
        This function trains the D and the G
        Args:
            batch_size:
                batch size
            index:
                index
            scaled_data (np.array):
                transformed input data

        Returns:
            generator and discriminator losses
        """
        # We freeze the discriminator's layers since we're
        # only interested in the generator and its learning
        self.D_model.trainable = True
        half_batch_size = int(batch_size * 0.5)
        # Create a batch of original data and train the model
        x_real,\
            y_real = self.get_input_samples(
                data=scaled_data,
                batch_size=half_batch_size,
                index=index
            )
        d_real_loss = self.D_model.train_on_batch(
            x_real,
            y_real,
            reset_metrics=True
        )

        # Create a batch of data generated by G and train
        # the model. We call data from G 'fake' data.
        # Runs a single gradient update on a single
        # batch of data.

        x_fake,\
            y_fake = self.create_fake_samples(
                batch_size=half_batch_size
            )
        d_fake_loss = self.D_model.train_on_batch(
            x_fake,
            y_fake,
            reset_metrics=True
        )
        avg_dis_loss = d_real_loss * 0.5 + d_fake_loss * 0.5

        # Create noise for the generator model. We will
        # draw samples from a uniform distribution
        noise = np.random.uniform(
            0,
            1,
            (batch_size, self.noise_dim[0])
        )

        gen_loss = self.wgan_model.train_on_batch(
            noise,
            np.ones((batch_size, 1))
        )
        return avg_dis_loss, gen_loss

    @staticmethod
    def get_input_samples(data, batch_size, index):
        """
        Generate batch_size of real samples with class labels
        Args:
            data:
                input data
            batch_size:
                bacth size
            index:
                index

        Returns:
            data sample array and labels array
        """
        idx_0 = batch_size * index
        idx_z = idx_0 + batch_size
        samples = data[idx_0: idx_z]

        return samples, np.ones((len(samples), 1))

    def create_fake_samples(self, batch_size):
        """
        Use the generator to generate batch_size
        fake examples.
        Args:
            batch_size:
                bacth size

        Returns:
            fake data array  and labels array
        """
        noise = np.random.uniform(
            0, 1,
            (batch_size, self.noise_dim[0])
        )
        x = self.G_model.predict(noise)
        return x, np.zeros((len(x), 1))

    if __name__ == '__main__':
        build_generator()
        build_discriminator()
        build_wgan()
        train()
        train_models()
        get_input_samples()
        create_fake_samples()
