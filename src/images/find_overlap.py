import argparse
from random import random

import tensorflow as tf
import numpy as np

import config.images.config as cfg

from config import paths
from src.images import preprocessing, slide_info
import os
import shutil
from skimage.transform import resize

max_overlap = 224
my_seed = 42


def find_overlap(overlap=0):
    np.random.seed(my_seed)
    tf.random.set_seed(my_seed)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    normal_slides_info, tumor_slides_info = slide_info.read_slides_info()

    if paths.selected_tiles_dir_with_overlap.exists():
        shutil.rmtree(paths.selected_tiles_dir_with_overlap)

    if not paths.selected_tiles_dir_with_overlap.exists():
        os.mkdir(paths.selected_tiles_dir_with_overlap)

    slides_info = normal_slides_info[:30] + tumor_slides_info[:30]

    preprocessing.extract_tiles_on_disk(
        slides_info=slides_info,
        overlap=overlap,
        root_save_path=paths.selected_tiles_dir_with_overlap
    )

    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

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
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(500, activation='relu')(x)
    # x = NoisyAnd(1)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # nuovo
    # x = tf.keras.layers.Dense(2000, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # x = tf.keras.layers.Dense(2, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=base_model.inputs, outputs=x)

    files = list(os.listdir(paths.selected_tiles_dir_with_overlap))
    labels = list(map(lambda x: int(x.split("_")[2].replace(".npy", "")), files))

    total_length = len(files)
    train_length = int(0.7 * total_length)

    batch_size = 32

    train_gen = create_generator(files[:train_length], labels[:train_length], batch_size=batch_size)
    val_gen = create_generator(files[-train_length:], labels[-train_length:], batch_size=batch_size)

    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath="cp4.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    opt = tf.keras.optimizers.Adam(learning_rate=0.000005)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=['accuracy']
    )

    epochs = 3

    model.fit(
        x=train_gen,
        steps_per_epoch=train_length // batch_size,
        validation_data=val_gen,
        validation_steps=(total_length - train_length) // batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[check_point],
        verbose=True
    )


def create_generator(files, labels, batch_size=64):
    while True:
        out_data = []
        out_labels = []
        for i, (file, label) in enumerate(zip(files, labels)):
            img = np.load(paths.selected_tiles_dir_with_overlap / file)
            # img = resize(img, (224, 224))
            img = img / 255.0
            out_data.append(img)
            out_labels.append(label)

            while len(out_data) > batch_size:
                yield np.array(out_data[:batch_size]), np.array(out_labels[:batch_size]).reshape(-1, 1)
                out_data = out_data[:batch_size]
                out_labels = out_labels[:batch_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--overlap',
                        help='Overlap value',
                        required=True,
                        type=int)

    args = parser.parse_args()

    #find_overlap(overlap=0)
    #find_overlap(overlap=1)
    find_overlap(overlap=2)
    find_overlap(overlap=10)
    find_overlap(overlap=100)
