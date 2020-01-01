from typing import Optional, Tuple

from keras.engine import Layer
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization, MaxPooling2D, Activation
from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Optimizer, Adam


class GenerativeNetwork():

    def __init__(self, optimizer: Optimizer = Adam(0.0002, 0.5)):
        self.optimizer = optimizer

    def conv2d(self, initial_layer: Layer, filters: int, kernel_size: int = (4, 4), strides: int = (2, 2),
               momentum: Optional[float] = 0.99, padding: str = 'same', lrelu_alpha=0.2):
        down_conv = Conv2D(filters, kernel_size, strides=strides, padding=padding,
                           kernel_initializer=RandomNormal(stddev=0.02))(initial_layer)
        down_conv = BatchNormalization(momentum=momentum)(down_conv) if momentum else down_conv
        down_conv = LeakyReLU(alpha=lrelu_alpha)(down_conv)
        return down_conv

    def intermediate(self, initial_layer: Layer, filters: int, kernel_size: int = (4, 4), strides: int = (2, 2)):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same')(initial_layer)
        conv = Activation('relu')(conv)
        return conv

    def deconv2d(self, initial_layer: Layer, skipped_layer: Layer, filters: int, kernel_size=(4, 4),
                 padding: str = 'same', dropout_rate: Optional[float] = 0.5, strides=(2, 2), activation='relu',
                 momentum: Optional[float] = 0.99):
        up_conv = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
                                  kernel_initializer=RandomNormal(stddev=0.02))(initial_layer)
        up_conv = BatchNormalization(momentum=momentum)(up_conv) if momentum else up_conv
        up_conv = Dropout(dropout_rate)(up_conv, training=True) if dropout_rate else up_conv
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

        d1 = self.conv2d(input, init_filters, momentum=None)
        d2 = self.conv2d(d1, init_filters * 2)
        d3 = self.conv2d(d2, init_filters * 4)
        d4 = self.conv2d(d3, init_filters * 8)
        d5 = self.conv2d(d4, init_filters * 8)
        d6 = self.conv2d(d5, init_filters * 8)
        d7 = self.conv2d(d6, init_filters * 8)

        intermediate = self.intermediate(d7, init_filters * 8)

        u1 = self.deconv2d(intermediate, d7, init_filters * 8, dropout_rate=dropout_rate)
        u2 = self.deconv2d(u1, d6, init_filters * 8, dropout_rate=dropout_rate)
        u3 = self.deconv2d(u2, d5, init_filters * 8, dropout_rate=dropout_rate)
        u4 = self.deconv2d(u3, d4, init_filters * 8, dropout_rate=None)
        u5 = self.deconv2d(u4, d3, init_filters * 4, dropout_rate=None)
        u6 = self.deconv2d(u5, d2, init_filters * 2, dropout_rate=None)
        u7 = self.deconv2d(u6, d1, init_filters, dropout_rate=None)

        output = Conv2DTranspose(output_channels, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                 activation=output_activation)(u7)

        model = Model(input, output, name='generator')
        if compile:
            model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

        return model
