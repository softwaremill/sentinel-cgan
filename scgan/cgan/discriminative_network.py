from typing import Optional, Tuple

from keras.engine import Layer
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization, GaussianNoise
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Optimizer, Adam


class DiscriminativeNetwork():

    def __init__(self, optimizer: Optimizer = Adam(0.0002, 0.5)):
        self.optimizer = optimizer

    def conv2d(self, initial_layer: Layer, filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int],
               lrelu_alpha: float, momentum: Optional[float]):
        conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                      kernel_initializer=RandomNormal(stddev=0.02))(initial_layer)
        conv = BatchNormalization(momentum=momentum)(conv) if momentum else conv
        conv = LeakyReLU(alpha=lrelu_alpha)(conv)
        return conv

    def build(self, init_filters: int,
              input_shape: Tuple[int, int, int],
              condition_shape: Tuple[int, int, int],
              kernel_size: Tuple[int, int] = (4, 4),
              strides: Tuple[int, int] = (2, 2),
              lrelu_alpha: float = 0.2,
              momentum: Optional[float] = 0.99,
              noise_stdev: float = 0.01,
              compile: bool = True) -> Model:
        target = Input(shape=input_shape, name='satellite_image')
        noise = GaussianNoise(noise_stdev)(target)
        condition = Input(shape=condition_shape, name='condition_mask')
        combination = Concatenate(axis=-1)([noise, condition])

        d = self.conv2d(combination, init_filters, kernel_size, strides, lrelu_alpha, momentum=None)
        d = self.conv2d(d, init_filters * 2, kernel_size, strides, lrelu_alpha, momentum)
        d = self.conv2d(d, init_filters * 4, kernel_size, strides, lrelu_alpha, momentum)
        d = self.conv2d(d, init_filters * 8, kernel_size, strides, lrelu_alpha, momentum)
        output = Conv2D(1, kernel_size=kernel_size, strides=strides, padding='same', activation='sigmoid')(d)

        model = Model([target, condition], output, name='discriminator')
        if compile:
            model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'],
                          loss_weights=[0.5])

        return model
