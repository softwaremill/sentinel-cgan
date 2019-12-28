import os
from datetime import datetime

import matplotlib.gridspec as gridspec
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

        print('Plotter has been created (dir: %s)' % self.out_dir)

    def plot_epoch_result(self, epoch: int, channels: int = 4):
        samples_count = len(self.data_generator.images_df(Purpose.PLOT))

        test_satellite_imgs, test_masks = next(self.data_generator.load(samples_count, Purpose.PLOT, random_state=0))
        predicted_satellite_images = self.model.predict(test_masks)

        fig = plt.figure(figsize=(channels + 2, samples_count + 1))
        spec = gridspec.GridSpec(samples_count, channels + 1, fig, wspace=0.075, hspace=0.075,
                                 top=1.0 - 0.5 / (samples_count + 1), bottom=0.5 / (samples_count + 1),
                                 left=0.5 / (channels + 2), right=1 - 0.5 / (channels + 2))

        for row in range(samples_count):
            for column in range(channels + 1):

                ax = fig.add_subplot(spec[row, column])
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                if column < channels:
                    ax.imshow(predicted_satellite_images[row][:, :, column], cmap='pink')
                    if row == 0:
                        ax.set_title('channel %s' % column)
                else:
                    ax.imshow(test_masks[row][:, :, 0], cmap='gray')
                    if row == 0:
                        ax.set_title('mask')

        plt.savefig('%s/epoch_%s.png' % (self.out_dir, epoch))
        plt.close()

    def plot_history(self, history: History):
        plt.figure(figsize=(14, 7))

        plt.plot(history.history['discriminator_avg_acc'])
        plt.plot(history.history['discriminator_artificial_acc'])
        plt.plot(history.history['discriminator_real_acc'])
        plt.title('GAN accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator accuracy (avg)', 'Discriminator accuracy (artificial)',
                    'Discriminator accuracy (real)'], loc='upper left')
        plt.savefig('%s/accuracy.png' % self.out_dir)

        plt.clf()

        plt.plot(history.history['discriminator_avg_loss'])
        plt.plot(history.history['discriminator_artificial_loss'])
        plt.plot(history.history['discriminator_real_loss'])
        plt.plot(history.history['generator_loss'])
        plt.title('GAN loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator loss (avg)', 'Discriminator loss (artificial)',
                    'Discriminator loss (real)', 'Generator loss'], loc='upper left')
        plt.savefig('%s/loss.png' % self.out_dir)

        plt.close()
