from keras import Model
from keras.models import load_model

from data.data_generator import SentinelDataGenerator
from util.plotter import Plotter

if __name__ == '__main__':
    generator_model: Model = load_model('../model/generator.h5')
    generator_model.summary()

    data_generator = SentinelDataGenerator('bdot', satellite_image_shape=(4, 256, 256),
                                           landcover_mask_shape=(1, 256, 256), feature_range=(-1, 1))
    plotter = Plotter(generator_model, data_generator, sub_dir='predict')

    plotter.predict_and_plot_images(batch=10)
    print('Finished prediction using %s model' % generator_model.name)
