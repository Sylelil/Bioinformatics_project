import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import config.images.config as cfg
from tensorflow.keras.layers import Layer
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from random import random


class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        self.b = self.add_weight(name='b',
                                 shape=(1, input_shape[3]),
                                 initializer='uniform',
                                 trainable=True)
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (
                tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]

    def from_config(self):
        return NoisyAnd(2)


def cnn_classifier():
    base_model = ResNet50(weights='imagenet', include_top=False)

    # non effettuiamo il freeze dell'ultimo strato convoluzionale

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-cfg.NUM_TRAINABLE_LAYERS:]:
        if not layer.name.endswith("bn"):
            layer.trainable = True

    x = base_model.output
    # x = tf.keras.layers.Dense(2000, activation='relu')(x)
    # x = tf.keras.layers.Dense(1000, activation='relu')(x)
    # x = tf.keras.layers.Dense(500, activation='relu')(x)
    # x = NoisyAnd(2)(x)
    # x = tf.keras.layers.Dense(2, activation='sigmoid')(x)

    # v2
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = NoisyAnd(1)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # nuovo
    # x = tf.keras.layers.Dense(2000, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(2, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=x)

    # if Path("cp3.h5").exists():
    #   model = tf.keras.models.load_model(Path("cp.h5"), custom_objects={"NoisyAnd": NoisyAnd})

    files = list(os.listdir(Path("D:\\Bioinformatics_project\\results\\images\\selected_tiles\\tiles")))

    ## down sampling
    # files = list(filter(lambda x: x.endswith("0.npy") or random() < 0.1, files))
    ## end

    # print(files[:50])

    labels = list(map(lambda x: int(x.split("_")[2].replace(".npy", "")), files))

    total_length = len(files)
    train_length = int(0.7 * total_length)

    batch_size = 64

    train_gen = create_generator(files[:train_length], labels[:train_length], batch_size=batch_size)
    val_gen = create_generator(files[-train_length:], labels[-train_length:], batch_size=batch_size)

    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath="cp2.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    for layer in base_model.layers:
        print("{}: {}".format(layer.name, layer.trainable))

    opt = tf.keras.optimizers.Adam(learning_rate=0.000005)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=['accuracy']
    )

    epochs = 10

    history = model.fit(
        x=train_gen,
        steps_per_epoch=train_length // batch_size,
        validation_data=val_gen,
        validation_steps=(total_length - train_length) // batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[check_point],
        verbose=True
    )

    model.save(cfg.FINE_TUNED_MODEL_NAME)

    plot_training(history, epochs)

    print(model.summary())


def create_generator(files, labels, batch_size=64):
    while True:
        out_data = []
        out_labels = []
        for i, (file, label) in enumerate(zip(files, labels)):
            img = np.load(Path("D:\\Bioinformatics_project\\results\\images\\selected_tiles\\tiles") / file)
            img = img / 255.0
            out_data.append(img)
            out_labels.append(label)

            if int(label) == 0:
                for _ in range(8):
                    # Facciamo attenzione a non introdurre alterazioni del colore
                    # (saturazione, contrasto, ecc) perché, essendo sbilanciati, rischierebbero
                    # di specializzare la rete neurale su queste alterazioni
                    data_augmentation = tf.keras.Sequential([
                        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                        tf.keras.layers.experimental.preprocessing.RandomZoom(0.4),
                    ])

                    augmented_tile = data_augmentation(np.expand_dims(img, axis=0))
                    augmented_tile_npy = np.squeeze(augmented_tile.numpy(), axis=0)
                    out_data.append(augmented_tile_npy)
                    out_labels.append(label)

            while len(out_data) > batch_size:
                yield np.array(out_data[:batch_size]), np.array(out_labels[:batch_size]).reshape(-1, 1)
                out_data = out_data[:batch_size]
                out_labels = out_labels[:batch_size]
            # if len(out_data) == batch_size:
            #     yield np.array(out_data), np.array(out_labels).reshape(-1, 1)


def plot_training(H, N):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    cnn_classifier()
