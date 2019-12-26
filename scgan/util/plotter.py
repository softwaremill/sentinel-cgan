from datetime import time, datetime

from keras import Model

from data.generator import Purpose, SentinelDataGenerator
import matplotlib.pyplot as plt
import os


class Plotter:

    def __init__(self, model: Model, data_generator: SentinelDataGenerator):
        self.model = model
        self.data_generator = data_generator
        self.out_dir = '../figs/out/%s' % datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        os.mkdir(self.out_dir)

    def plot(self, epoch_number: int, channels: int = 4):

        test_satellite_imgs, test_masks = next(self.data_generator.load(1, Purpose.TEST))
        predicted_satellite_images = self.model.predict(test_masks)

        fig, axs = plt.subplots(1, 5, figsize=(21, 7))
        for index, ax in enumerate(axs):
            if index < channels:
                ax.imshow(predicted_satellite_images[0][:, :, index], cmap='pink')
                ax.set_title('channel %s' % index)
            else:
                ax.imshow(test_masks[0][:, :, 0], cmap='pink')
                ax.set_title('mask')

        plt.savefig('%s/epoch_%s.png' % (self.out_dir, epoch_number))
