import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_train_data(lookup_dir):
    features_list = []

    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        data = np.load(file_path)
        np_features_list = list(data)
        features_list.extend(np_features_list)

    return np.array(features_list)


def main():
    normal_images_save_dir = Path('generated') / 'numpy_normal'
    tumor_images_save_dir = Path('generated') / 'numpy_tumor'

    if not os.path.exists(Path('generated')):
        print("%s not existing." % Path('generated'))
        exit()
    if not os.path.exists(normal_images_save_dir):
        print("%s not existing." % normal_images_save_dir)
        exit()
    if not os.path.exists(tumor_images_save_dir):
        print("%s not existing." % tumor_images_save_dir)
        exit()

    # features:
    normal_train_features = get_train_data(normal_images_save_dir)
    normal_labels_list = np.zeros((len(normal_train_features),), dtype=int)

    tumor_train_features = get_train_data(tumor_images_save_dir)
    tumor_labels_list = np.ones((len(tumor_train_features),), dtype=int)

    train_features = np.concatenate((normal_train_features, tumor_train_features))
    train_labels = np.concatenate((normal_labels_list, tumor_labels_list))

    print(train_features.shape)
    print(train_labels.shape)

    # creare il modello
    model = keras.Sequential([
        # keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_features, train_labels, epochs=10, shuffle=True, validation_split=0.5)


if __name__ == '__main__':
    main()