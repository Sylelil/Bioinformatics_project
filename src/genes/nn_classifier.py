import datetime
import os

import tensorflow as tf


class MultiLayerPerceptron:

    def __init__(self,
                 n_epochs,
                 batch_size,
                 learning_rate,
                 num_features,
                 n_classes,
                 logdir=None,
                 lr_reduction_epoch=25):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.n_classes = n_classes
        self.lr_reduction_epoch = lr_reduction_epoch

        if logdir is None:
            self.logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        self._build_model()

    def _build_model(self):
        # shallow model - mlp
        self.model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(100, input_dim=self.num_features, activation='relu', name="fc1"),
                    tf.keras.layers.Dense(50, activation='relu', name="fc2"),
                    tf.keras.layers.Dense(self.n_classes, activation='softmax', name="predictions")
        ])

        print("Model built")

    def train_model(self, X_train, X_val, y_val, y_train):
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

        self.model.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(self.logdir,
                                                              histogram_freq=1,
                                                              write_grads=True,
                                                              update_freq='epoch')
        scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler,
                                                                      verbose=1)
        self.history = self.model.fit(x=X_train,
                                      y=y_train,
                                      epochs=self.n_epochs,
                                      batch_size=self.batch_size,
                                      validation_data=(X_val, y_val),
                                      callbacks=[tensorboard_callback,
                                                 tf.keras.callbacks.History(),
                                                 scheduler_callback])

    def lr_scheduler(self, epoch, lr):
        if epoch == self.lr_reduction_epoch:
            return lr * 0.1
        else:
            return lr
