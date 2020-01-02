import os
from datetime import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.callbacks import History

from data.data_generator import Purpose, DataGenerator


class Plotter:

    def __init__(self, model: Model, data_generator: DataGenerator):
        self.model = model
        self.data_generator = data_generator
        self.out_dir = Path(
            '../figs/out/%s/%s' % (data_generator.dataset, datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f'))
        ).resolve()
        os.makedirs(self.out_dir)

        print('Plotter has been created (dir: %s)' % self.out_dir)

    def plot_epoch_result(self, epoch: int, channels: int = 4):
        print('\nPlotting epoch %s results' % (epoch + 1))

        samples_count = len(self.data_generator.images_df(Purpose.PLOT))

        test_satellite_images, test_masks = next(self.data_generator.load(samples_count, Purpose.PLOT, random_state=0))
        predicted_satellite_images = self.model.predict(test_masks)

        fig = plt.figure(figsize=(channels * 2 + 2, samples_count + 1))
        spec = gridspec.GridSpec(samples_count, channels * 2 + 1, fig, wspace=0.075, hspace=0.075,
                                 top=1.0 - 0.5 / (samples_count + 1), bottom=0.5 / (samples_count + 1),
                                 left=0.5 / (channels * 2 + 2), right=1 - 0.5 / (channels * 2 + 2))

        for row in range(samples_count):
            next_col_position = 0
            for column in range(channels):
                ax = fig.add_subplot(spec[row, next_col_position])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                next_col_position += 1

                ax.imshow(predicted_satellite_images[row][:, :, column], cmap='pink')
                if row == 0:
                    ax.set_title('artificial (%s)' % column)

                ax = fig.add_subplot(spec[row, next_col_position])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                next_col_position += 1

                ax.imshow(test_satellite_images[row][:, :, column], cmap='pink')
                if row == 0:
                    ax.set_title('real (%s)' % column)

            ax = fig.add_subplot(spec[row, next_col_position])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if test_masks[row].shape[2] == 3:
                ax.imshow(test_masks[row][:, :, 0:3])
            else:
                ax.imshow(test_masks[row][:, :, 0], cmap='pink')

            if row == 0:
                ax.set_title('mask')

        plt.savefig(Path('%s/epoch_%s.png' % (self.out_dir, epoch + 1)))
        plt.close()

    def plot_history(self, history: History):
        plt.figure(figsize=(14, 7))

        daa = history.history['discriminator_artificial_acc']
        dra = history.history['discriminator_real_acc']

        plt.plot(daa)
        plt.plot(dra)
        plt.fill_between(np.arange(history.params['epochs']), daa, dra, alpha=0.3, facecolor='grey')
        plt.title('GAN accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator accuracy (artificial)', 'Discriminator accuracy (real)',
                    'Discriminator accuracy (diff)'], loc='upper left')
        plt.savefig(Path('%s/accuracy.png' % self.out_dir))

        plt.clf()

        plt.plot(history.history['discriminator_artificial_loss'])
        plt.plot(history.history['discriminator_real_loss'])
        plt.plot(history.history['generator_loss'])
        plt.title('GAN loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator loss (artificial)', 'Discriminator loss (real)', 'Generator loss'], loc='upper left')
        plt.savefig(Path('%s/loss.png' % self.out_dir))

        plt.close()
