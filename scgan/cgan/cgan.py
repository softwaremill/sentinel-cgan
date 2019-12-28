from typing import Tuple, List

import numpy as np
from keras import Model, Input
from keras.callbacks import History, BaseLogger, ProgbarLogger, CallbackList, Callback
from keras.optimizers import Adam, Optimizer

from data.generator import SentinelDataGenerator
from util.plotter import Plotter


class CGAN():

    def __init__(self, data_generator: SentinelDataGenerator,
                 discriminative_network_model: Model,
                 generative_network_model: Model,
                 input_shape: Tuple[int, int, int],
                 condition_shape: Tuple[int, int, int],
                 optimizer: Optimizer = Adam(0.0005, 0.5)):
        self.data_generator = data_generator
        self.discriminative_network_model = discriminative_network_model
        self.generative_network_model = generative_network_model
        self.input_shape = input_shape
        self.condition_shape = condition_shape

        input = Input(shape=input_shape)
        condition = Input(shape=condition_shape)
        artificial = self.generative_network_model(condition)
        frozen_discriminative_network_model = Model(inputs=discriminative_network_model.inputs,
                                                    outputs=discriminative_network_model.outputs)
        frozen_discriminative_network_model.trainable = False
        validatable = frozen_discriminative_network_model([artificial, condition])

        self.cgan_model = Model(inputs=[input, condition], outputs=[validatable, artificial], name='sentinel-cgan')
        self.cgan_model.compile(loss=['mae', 'mse'], optimizer=optimizer)
        self.cgan_model.stop_training = False
        self.plotter = Plotter(generative_network_model, data_generator)

    def fit(self, epochs: int = 1, batch: int = 1, pixel_range: Tuple[int, int] = (0, 1),
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

            for i, (satellite_images, mask_images) in enumerate(self.data_generator.load(batch)):
                effective_batch_size = len(satellite_images)

                batch_logs = {'batch': i, 'size': effective_batch_size}
                callbacks.on_batch_begin(i, batch_logs)

                def form_base(bound):
                    modifier = int(self.input_shape[0] / 2 ** 4)
                    return np.full((effective_batch_size,) + (modifier, modifier, 1), bound)

                artificial_base = form_base(pixel_range[0])
                validatable_base = form_base(pixel_range[1])

                artificial_satellite_image = self.generative_network_model.predict(mask_images)

                real_dn_loss = self.discriminative_network_model.train_on_batch(x=[satellite_images, mask_images],
                                                                                y=validatable_base)

                artificial_dn_loss = self.discriminative_network_model.train_on_batch(
                    x=[artificial_satellite_image, mask_images],
                    y=artificial_base)

                gn_loss = self.cgan_model.train_on_batch(x=[satellite_images, mask_images],
                                                         y=[validatable_base, satellite_images])

                epoch_logs.update({
                    'discriminator_artificial_acc': artificial_dn_loss[1],
                    'discriminator_artificial_loss': artificial_dn_loss[0],
                    'discriminator_real_acc': real_dn_loss[1],
                    'discriminator_real_loss': real_dn_loss[0],
                    'generator_loss': gn_loss[0]
                })

                callbacks.on_batch_end(i)
                if self.cgan_model.stop_training:
                    break

            self.plotter.plot_epoch_result(epoch)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.cgan_model.stop_training:
                break

        callbacks.on_train_end()
        self.plotter.plot_history(history)
        return history
