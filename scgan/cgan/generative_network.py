from typing import Optional, Tuple

from keras.engine import Layer
from keras.layers import BatchNormalization, MaxPooling2D, Activation
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Optimizer, Adam


class GenerativeNetwork():

    def __init__(self, optimizer: Optimizer = Adam(0.0002, 0.5)):
        self.optimizer = optimizer

    def conv2d(self, initial_layer: Layer, filters: int, kernel_size: int = (3, 3), strides: int = (2, 2),
               padding: str = 'same', dropout_rate: Optional[float] = None, activation='relu'):
        down_conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(initial_layer)
        down_conv = BatchNormalization()(down_conv)
        down_conv = LeakyReLU(alpha=0.2)(down_conv)
        return down_conv

    def deconv2d(self, initial_layer: Layer, skipped_layer: Layer, filters: int, kernel_size=(3, 3),
                 padding: str = 'same', dropout_rate: Optional[float] = None, strides=(1, 1), activation='relu'):
        up_conv = UpSampling2D(size=(2, 2))(initial_layer)
        up_conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, activation=activation)(up_conv)
        up_conv = BatchNormalization()(up_conv)
        up_conv = Dropout(dropout_rate)(up_conv) if dropout_rate else up_conv
        up_conv = Concatenate()([up_conv, skipped_layer])
        up_conv = Activation(activation)(up_conv)
        return up_conv

    def build(self, init_filters: int,
              input_shape: Tuple[int, int, int],
              output_channels: int,
              dropout_rate: Optional[float] = None,
              output_activation='tanh',
              compile: bool = True) -> Model:
        input = Input(shape=input_shape)

        d1 = self.conv2d(input, init_filters)
        d2 = self.conv2d(d1, init_filters * 2, dropout_rate=dropout_rate)
        d3 = self.conv2d(d2, init_filters * 4, dropout_rate=dropout_rate)
        d4 = self.conv2d(d3, init_filters * 8, dropout_rate=dropout_rate)
        d5 = self.conv2d(d4, init_filters * 8, dropout_rate=dropout_rate)
        d6 = self.conv2d(d5, init_filters * 8, dropout_rate=dropout_rate)
        d7 = self.conv2d(d6, init_filters * 8, dropout_rate=dropout_rate)

        u1 = self.deconv2d(d7, d6, init_filters * 8, dropout_rate=dropout_rate)
        u2 = self.deconv2d(u1, d5, init_filters * 8, dropout_rate=dropout_rate)
        u3 = self.deconv2d(u2, d4, init_filters * 8, dropout_rate=dropout_rate)
        u4 = self.deconv2d(u3, d3, init_filters * 4, dropout_rate=dropout_rate)
        u5 = self.deconv2d(u4, d2, init_filters * 2, dropout_rate=dropout_rate)
        u6 = self.deconv2d(u5, d1, init_filters)
        u7 = UpSampling2D(size=(2, 2))(u6)

        output = Conv2D(output_channels, kernel_size=(3,3), strides=(1, 1), padding='same', activation=output_activation)(u7)
        model = Model(input, output, name='generator')

        if compile:
            model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

        return model
