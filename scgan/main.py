import os

from rasterio.enums import Resampling

from cgan.cgan import CGAN
from cgan.discriminative_network import DiscriminativeNetwork
from cgan.generative_network import GenerativeNetwork
from data.generator import SentinelDataGenerator

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    satellite_image_shape = (512, 512, 4)
    mask_shape = (512, 512, 1)
    output_channels = satellite_image_shape[2]
    init_filters = 16

    data_generator = SentinelDataGenerator('corine', satellite_image_shape=(4, 512, 512),
                                           landcover_mask_shape=(1, 512, 512), feature_range=(-1, 1))
    dn = DiscriminativeNetwork()
    dn_model = dn.build(init_filters=init_filters, input_shape=satellite_image_shape, condition_shape=mask_shape)

    gn = GenerativeNetwork()
    gn_model = gn.build(init_filters=init_filters, input_shape=mask_shape, output_channels=output_channels,
                        compile=False, dropout_rate=0.5)
    cgan = CGAN(data_generator, dn_model, gn_model, input_shape=satellite_image_shape, condition_shape=mask_shape)

    history = cgan.fit(epochs=5, batch=5)

    print('Sentinel CGAN has been fitted')
