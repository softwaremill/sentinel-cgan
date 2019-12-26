from cgan.cgan import CGAN
from cgan.discriminative_network import DiscriminativeNetwork
from cgan.generative_network import GenerativeNetwork
from data.generator import SentinelDataGenerator, Purpose
import matplotlib.pyplot as plt

if __name__ == '__main__':

    satellite_image_shape = (128, 128, 4)
    mask_shape = (128, 128, 1)
    output_channels = 4
    init_filters = 32

    data_generator = SentinelDataGenerator('sample')
    dn = DiscriminativeNetwork()
    dn_model = dn.build(init_filters=init_filters, input_shape=satellite_image_shape, condition_shape=mask_shape)

    gn = GenerativeNetwork()
    gn_model = gn.build(init_filters=init_filters, input_shape=mask_shape, output_channels=output_channels)
    cgan = CGAN(data_generator, dn_model, gn_model, input_shape=satellite_image_shape, condition_shape=mask_shape)

    cgan.train(epochs=20, batch=32)