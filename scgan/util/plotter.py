import os
from datetime import datetime

import matplotlib.pyplot as plt
from keras import Model
from keras.callbacks import History

from data.generator import Purpose, SentinelDataGenerator


class Plotter:

    def __init__(self, model: Model, data_generator: SentinelDataGenerator):
        self.model = model
        self.data_generator = data_generator
        self.out_dir = '../figs/out/%s' % datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        os.mkdir(self.out_dir)

    def plot_epoch_result(self, epoch_number: int, channels: int = 4):
        plt.clf()

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

    def plot_history(self, history: History):
        plt.clf()

        plt.plot(history.history['discriminator_acc'])
        plt.plot(history.history['generator_acc'])
        plt.title('GAN accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator accuracy', 'Generator accuracy'], loc='upper left')
        plt.savefig('%s/accuracy.png' % self.out_dir)

        plt.clf()
        plt.plot(history.history['discriminator_loss'])
        plt.plot(history.history['generator_loss'])
        plt.title('GAN loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator loss', 'Generator loss'], loc='upper left')
        plt.savefig('%s/loss.png' % self.out_dir)
