import os
import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_train_data(lookup_dir):
    features_list = []
    for np_file in os.listdir(lookup_dir):
        data = np.load(np_file)
        features_list.append(data)
    return features_list

def main():
    dir_normal = Path('generated')
    dir_tumor = Path('generated')

    # features:
    train_features_normal = get_train_data()
    train_labels_normal = [0]*len(train_features_normal)
    train_features_tumor = get_train_data()
    train_labels_tumor = [0]*len(train_features_tumor)


    # creare il modello
    model = keras.Sequential([
        # keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_features, train_labels, epochs=10)

def if __name__ == '__main__':
    main()