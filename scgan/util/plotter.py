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

        print('Plotter has been created (dir: %s)' % self.out_dir)

    def plot_epoch_result(self, epoch: int, channels: int = 4):
        plt.clf()

        all_images_count = len(self.data_generator.images_df(Purpose.PLOT))

        test_satellite_imgs, test_masks = next(self.data_generator.load(all_images_count, Purpose.PLOT, random_state=0))
        predicted_satellite_images = self.model.predict(test_masks)

        fig, axs = plt.subplots(all_images_count, 5, figsize=(21, 21))
        for row, row_ax in enumerate(axs):
            for column, column_ax in enumerate(row_ax):
                if column < channels:
                    column_ax.imshow(predicted_satellite_images[row][:, :, column], cmap='pink')
                    column_ax.set_title('channel %s' % column)
                else:
                    column_ax.imshow(test_masks[row][:, :, 0], cmap='pink')
                    column_ax.set_title('mask')

        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        plt.savefig('%s/epoch_%s.png' % (self.out_dir, epoch))

    def plot_history(self, history: History):
        plt.clf()

        plt.figure(figsize=(14, 7))
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
