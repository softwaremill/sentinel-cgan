from scgan.cgan.cgan import CGAN
from scgan.cgan.discriminative_network import DiscriminativeNetwork
from scgan.cgan.generative_network import GenerativeNetwork
from scgan.data.data_generator import SentinelDataGenerator

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    satellite_image_shape = (256, 256, 4)
    mask_shape = (256, 256, 1)
    output_channels = satellite_image_shape[2]
    init_filters = 64

    data_generator = SentinelDataGenerator('bdot', satellite_image_shape=(4, 256, 256),
                                           landcover_mask_shape=(1, 256, 256), feature_range=(-1, 1))
    dn = DiscriminativeNetwork()
    dn_model = dn.build(init_filters=init_filters, input_shape=satellite_image_shape, condition_shape=mask_shape,
                        kernel_size=(3, 3))

    gn = GenerativeNetwork()
    gn_model = gn.build(init_filters=init_filters, input_shape=mask_shape, output_channels=output_channels,
                        compile=False, dropout_rate=0.5, kernel_size=(7, 7))
    cgan = CGAN(data_generator, dn_model, gn_model, input_shape=satellite_image_shape, condition_shape=mask_shape)

    history = cgan.fit(epochs=250, batch=20)

    print('Sentinel CGAN has been fitted')
