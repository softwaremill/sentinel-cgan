from typing import Optional, Tuple

from keras.layers import BatchNormalization, Dropout
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import Optimizer, Adam


class DiscriminativeNetwork():

    def __init__(self, optimizer: Optimizer = Adam(0.0002, 0.5)):
        self.optimizer = optimizer

    def conv2d(self, initial_layer, filters, kernel_size=(4, 4), strides=(2, 2), momentum: Optional[float] = None,
               activation='elu'):
        conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(
            initial_layer)
        conv = BatchNormalization(momentum=momentum)(conv) if momentum else conv
        return conv

    def build(self, init_filters: int,
              input_shape: Tuple[int, int, int],
              condition_shape: Tuple[int, int, int],
              momentum=0.8,
              compile: bool = True) -> Model:
        input = Input(shape=input_shape)
        condition = Input(shape=condition_shape)
        combination = Concatenate(axis=-1)([input, condition])

        d = self.conv2d(combination, init_filters)
        d = self.conv2d(d, init_filters * 2, momentum=momentum)
        d = self.conv2d(d, init_filters * 4, momentum=momentum)
        d = self.conv2d(d,  init_filters * 8, momentum=momentum)
        output = Conv2D(1, kernel_size=(3, 3), strides=1, padding='same', activation='sigmoid')(d)

        model = Model([input, condition], output, name='discriminator')

        if compile:
            model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

        return model
