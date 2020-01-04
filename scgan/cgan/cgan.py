from typing import Tuple, List

import numpy as np
from keras import Model, Input
from keras.callbacks import History, BaseLogger, ProgbarLogger, CallbackList, Callback
from keras.optimizers import Adam, Optimizer

from data.data_generator import DataGenerator
from util.plotter import Plotter


class CGAN():

    def __init__(self, data_generator: DataGenerator,
                 discriminative_network_model: Model,
                 generative_network_model: Model,
                 input_shape: Tuple[int, int, int],
                 condition_shape: Tuple[int, int, int],
                 optimizer: Optimizer = Adam(0.0002, 0.5)):
        self.data_generator = data_generator
        self.discriminative_network_model = discriminative_network_model
        self.generative_network_model = generative_network_model
        self.input_shape = input_shape
        self.condition_shape = condition_shape

        condition = Input(shape=condition_shape)
        artificial = self.generative_network_model(condition)
        frozen_discriminative_network_model = Model(
            inputs=discriminative_network_model.inputs,
            outputs=discriminative_network_model.outputs
        )
        frozen_discriminative_network_model.trainable = False
        discrimination_result = frozen_discriminative_network_model([artificial, condition])

        self.cgan_model = Model(
            inputs=[condition],
            outputs=[discrimination_result, artificial],
            name='sentinel-cgan'
        )
        self.cgan_model.compile(loss=['binary_crossentropy', 'mae'], optimizer=optimizer, loss_weights=[1, 100])
        self.cgan_model.stop_training = False
        self.plotter = Plotter(generative_network_model, data_generator)

    def fit(self, epochs: int = 1, batch: int = 1, artificial_label: int = 0, real_label: int = 1,
            callbacks: List[Callback] = None) -> History:

        processed_images_count = len(self.data_generator.images_df())

        callback_metrics = [
            'discriminator_artificial_acc', 'discriminator_artificial_loss',
            'discriminator_real_acc', 'discriminator_real_loss',
            'generator_loss'
        ]
        history = History()

        _callbacks = [
            BaseLogger(stateful_metrics=callback_metrics),
            ProgbarLogger(count_mode='steps', stateful_metrics=callback_metrics),
            history
        ]

        _callbacks = _callbacks + callbacks if callbacks else _callbacks

        callbacks = CallbackList(_callbacks)
        callbacks.set_model(self.cgan_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': int(processed_images_count / batch) + (processed_images_count % batch > 0),
            'samples': processed_images_count,
            'verbose': True,
            'do_validation': False,
            'metrics': callback_metrics
        })
        callbacks.on_train_begin()

        for epoch in range(epochs):

            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}

            epoch_artificial_dn_loss = []
            epoch_real_dn_loss = []
            epoch_gn_loss = []

            for i, (real_satellite_images, mask_images) in enumerate(self.data_generator.load(batch)):
                effective_batch_size = len(real_satellite_images)

                batch_logs = {'batch': i, 'size': effective_batch_size}
                callbacks.on_batch_begin(i, batch_logs)

                def form_base(bound):
                    modifier = int(self.input_shape[0] / 2 ** 5)
                    return np.full((effective_batch_size,) + (modifier, modifier, 1), bound)

                artificial_base = form_base(artificial_label)
                real_base = form_base(real_label)

                artificial_satellite_images = self.generative_network_model.predict(mask_images)

                batch_real_dn_loss = self.discriminative_network_model.train_on_batch(
                    x=[real_satellite_images, mask_images],
                    y=real_base
                )
                epoch_real_dn_loss.append(batch_real_dn_loss)

                batch_artificial_dn_loss = self.discriminative_network_model.train_on_batch(
                    x=[artificial_satellite_images, mask_images],
                    y=artificial_base
                )
                epoch_artificial_dn_loss.append(batch_artificial_dn_loss)

                batch_gn_loss = self.cgan_model.train_on_batch(
                    x=[mask_images],
                    y=[real_base, real_satellite_images]
                )
                epoch_gn_loss.append(batch_gn_loss)

                callbacks.on_batch_end(i)
                if self.cgan_model.stop_training:
                    break

            epoch_artificial_dn_loss = np.mean(epoch_artificial_dn_loss, axis=0)
            epoch_real_dn_loss = np.mean(epoch_real_dn_loss, axis=0)
            epoch_gn_loss = np.mean(epoch_gn_loss, axis=0)

            epoch_logs.update({
                'discriminator_artificial_acc': epoch_artificial_dn_loss[1],
                'discriminator_artificial_loss': epoch_artificial_dn_loss[0],
                'discriminator_real_acc': epoch_real_dn_loss[1],
                'discriminator_real_loss': epoch_real_dn_loss[0],
                'generator_loss': epoch_gn_loss[0]
            })

            self.plotter.plot_epoch_result(epoch, self.input_shape[2])
            callbacks.on_epoch_end(epoch, epoch_logs)
            self.plotter.plot_history(history)
            if self.cgan_model.stop_training:
                break

        callbacks.on_train_end()
        self.plotter.plot_history(history)
        return history
