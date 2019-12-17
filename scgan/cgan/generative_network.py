from typing import Optional, Tuple

from keras.engine import Layer
from keras.layers import BatchNormalization
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Optimizer, Adam


class GenerativeNetwork():

    def __init__(self, optimizer: Optimizer = Adam(0.0005, 0.5)):
        self.optimizer = optimizer

    def conv2d(self, initial_layer: Layer, filters: int, kernel_size: int = 4, strides: int = 2,
               momentum: Optional[float] = None):
        down_conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(initial_layer)
        down_conv = LeakyReLU(alpha=0.2)(down_conv)
        down_conv = BatchNormalization(momentum=momentum)(down_conv) if momentum else down_conv
        return down_conv

    def deconv2d(self, initial_layer: Layer, skipped_layer: Layer, filters: int, kernel_size=4,
                 dropout_rate: Optional[float] = None, strides=1, momentum: float = 0.8, activation='relu'):
        up_conv = UpSampling2D(size=2)(initial_layer)
        up_conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', activation=activation)(
            up_conv)
        up_conv = Dropout(dropout_rate)(up_conv) if dropout_rate else up_conv
        up_conv = BatchNormalization(momentum=momentum)(up_conv)
        up_conv = Concatenate()([up_conv, skipped_layer])
        return up_conv

    # TODO to init
    def build(self, init_filters: int,
              input_shape: Tuple[int, int, int],
              output_channels: int,
              batch_normalization_momentum=0.8,
              output_activation='tanh') -> Model:
        input = Input(shape=input_shape)

        d1 = self.conv2d(input, init_filters)
        d2 = self.conv2d(d1, init_filters * 2, momentum=batch_normalization_momentum)
        d3 = self.conv2d(d2, init_filters * 4, momentum=batch_normalization_momentum)
        d4 = self.conv2d(d3, init_filters * 8, momentum=batch_normalization_momentum)
        d5 = self.conv2d(d4, init_filters * 8, momentum=batch_normalization_momentum)
        d6 = self.conv2d(d5, init_filters * 8, momentum=batch_normalization_momentum)
        d7 = self.conv2d(d6, init_filters * 8, momentum=batch_normalization_momentum)

        u1 = self.deconv2d(d7, d6, init_filters * 8)
        u2 = self.deconv2d(u1, d5, init_filters * 8)
        u3 = self.deconv2d(u2, d4, init_filters * 8)
        u4 = self.deconv2d(u3, d3, init_filters * 4)
        u5 = self.deconv2d(u4, d2, init_filters * 2)
        u6 = self.deconv2d(u5, d1, init_filters)
        u7 = UpSampling2D(size=2)(u6)

        output = Conv2D(output_channels, kernel_size=4, strides=1, padding='same', activation=output_activation)(u7)
        model = Model(input, output)

        if compile:
            model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

        return model
