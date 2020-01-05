import os
from datetime import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from keras import Model
from keras.callbacks import History

from data.data_generator import Purpose, DataGenerator


class Plotter:

    def __init__(self, model: Model, data_generator: DataGenerator, sub_dir: str = 'train'):
        self.model = model
        self.data_generator = data_generator
        self.out_dir = Path(
            '../out/%s/%s/%s' % (sub_dir, data_generator.dataset, datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f'))
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

    def predict_and_plot_images(self, batch=1):
        real_satellite_images, mask_images = next(self.data_generator.load(batch, purpose=Purpose.TEST))
        artificial_satellite_images = self.model.predict(mask_images)

        def plot(image, image_id):
            channels = image.shape[2]
            fig = plt.figure(figsize=(channels + 1, 2))
            columns = channels + 1 if channels >= 3 else channels
            spec = gridspec.GridSpec(1, columns, fig, wspace=0.075, hspace=0.075,
                                     top=1.0 - 0.5 / 2, bottom=0.5 / 2,
                                     left=0.5 / (columns + 1), right=1 - 0.5 / (columns + 1))

            for column in range(channels):
                ax = fig.add_subplot(spec[0, column])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_title('ch: %s' % column)

                ax.imshow(image[:, :, column], cmap='pink')

            if channels >= 3:
                ax = fig.add_subplot(spec[0, channels])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_title('RGB')
                ax.imshow(((image[:, :, 0:3] + 1) * 127.5).astype(np.uint8))

            plt.savefig(Path('%s/%s.png' % (self.out_dir, image_id)))
            plt.close()

        with tqdm.tqdm(total=len(mask_images)) as t:
            for i, (real, artificial) in enumerate(zip(real_satellite_images, artificial_satellite_images)):
                plot(real, '%s_real' % i)
                plot(artificial, '%s_artificial' % i)
                t.update()

    def plot_history(self, history: History):
        plt.figure(figsize=(14, 7))

        daa = history.history['discriminator_artificial_acc']
        dra = history.history['discriminator_real_acc']

        plt.plot(daa)
        plt.plot(dra)
        plt.fill_between(np.arange(len(dra)), daa, dra, alpha=0.3, facecolor='grey')
        plt.title('GAN accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Discriminator accuracy (artificial)', 'Discriminator accuracy (real)',
                    'Discriminator accuracy (diff)'], loc='upper left')
        plt.savefig(Path('%s/_accuracy.png' % self.out_dir))

        plt.clf()

        color = 'tab:red'
        plt.plot(history.history['discriminator_artificial_loss'], color=color)
        plt.plot(history.history['discriminator_real_loss'], color=color)
        plt.tick_params(axis='y', labelcolor=color)
        plt.ylabel('Discriminator loss', color=color)
        plt.legend(['Discriminator loss (artificial)', 'Discriminator loss (real)'], loc='upper left')

        plt.twinx()

        color = 'tab:blue'
        plt.plot(history.history['generator_loss'], color=color)
        plt.tick_params(axis='y', labelcolor=color)
        plt.legend(['Generator loss'], loc='upper left')

        plt.title('GAN loss')
        plt.ylabel('Generator loss', color=color)
        plt.xlabel('Epoch')
        plt.savefig(Path('%s/_loss.png' % self.out_dir))

        plt.close()
