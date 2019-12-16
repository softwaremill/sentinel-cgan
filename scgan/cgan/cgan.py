from typing import Tuple

import numpy as np
from keras import Model, Input
from keras.optimizers import Adam, Optimizer

from data.generator import SentinelDataGenerator


class CGAN():
    data_generator: SentinelDataGenerator
    discriminative_network_model: Model
    generative_network_model: Model
    cgan_model: Model
    input_shape: Tuple[int, int, int]
    condition_shape: Tuple[int, int, int]

    def __init__(self, data_generator: SentinelDataGenerator,
                 discriminative_network_model: Model,
                 generative_network_model: Model,
                 input_shape: Tuple[int, int, int],
                 condition_shape: Tuple[int, int, int],
                 optimizer: Optimizer = Adam(0.0005, 0.5)):
        self.data_generator = data_generator
        self.discriminative_network_model = discriminative_network_model
        self.generative_network_model = generative_network_model
        self.input_shape = input_shape
        self.condition_shape = condition_shape

        input = Input(shape=input_shape)
        condition = Input(shape=condition_shape)
        artificial = self.generative_network_model(condition)
        validatable = self.discriminative_network_model([artificial, condition])

        self.cgan_model = Model(inputs=[input, condition], outputs=[validatable, artificial])
        self.cgan_model.compile(loss=['mae', 'mse'], optimizer=optimizer)

    # TODO log, accumulate information
    def train(self, epochs: int = 5, batch: int = 1, pixel_range: Tuple[int, int] = (0, 4096)):

        def form_base(bound):
            modifier = int(self.input_shape[0] / 2 ** 4)
            return np.full((batch,) + (modifier, modifier, 1), bound)

        artificial_base = form_base(pixel_range[0])
        validatable_base = form_base(pixel_range[1])

        for epoch in range(epochs):
            for i, (satellite_images, mask_images) in enumerate(self.data_generator.load(batch)):
                artificial_satellite_image = self.generative_network_model.predict(mask_images)

                real_dn_loss = self.discriminative_network_model.train_on_batch(x=[satellite_images, mask_images],
                                                                                y=validatable_base)

                artificial_dn_loss = self.discriminative_network_model.train_on_batch(
                    x=[artificial_satellite_image, mask_images],
                    y=artificial_base)

                dn_loss = np.mean([real_dn_loss, artificial_dn_loss])
                gn_loss = self.cgan_model.train_on_batch(x=[satellite_images, mask_images],
                                                         y=[artificial_base, satellite_images])

                print('[%s/%s/%s] -> discriminator loss: %s, generator: %s' % (epoch, epochs, i, dn_loss, gn_loss))
