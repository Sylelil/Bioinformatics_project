import argparse
import os
import sys
from os import path
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.common import class_balancing, feature_concatenation

def get_train_data(lookup_dir):
    features_list = []

    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        data = np.load(file_path)
        np_features_list = list(data)
        features_list.extend(np_features_list)

    return np.array(features_list)

'''
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

    
    #vgg16:
    # model = keras.Sequential([
    #     keras.layers.Dense(4096, activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(4096, activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(2, activation='softmax')
    # ])
    

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features, train_labels, epochs=10, shuffle=True, validation_split=0.5)
'''


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--method',
                        help='Feature extraction method',
                        choices=['fine_tuning', 'fixed_feature_generator'],
                        required=True,
                        type=str)
    parser.add_argument('--balancing',
                        help='Class balancing method',
                        choices=['random_upsampling', 'combined', 'smote', 'downsampling'],
                        required=False,
                        type=str)
    args = parser.parse_args()

    # Read tile features from file
    tile_features_train_dir = Path('results') / 'images' / 'extracted_features' / 'training'
    tile_features_test_dir = Path('results') / 'images' / 'extracted_features' / 'test'
    gene_features_train_dir = Path('results') / 'genes' / 'extracted_features' / 'training'
    gene_features_test_dir = Path('results') / 'genes' / 'extracted_features' / 'test'

    if not os.path.exists(Path('results')):
        print("%s not existing." % Path('results'))
        exit()
    if not os.path.exists(tile_features_train_dir):
        print("%s not existing." % tile_features_train_dir)
        exit()
    if not os.path.exists(tile_features_test_dir):
        print("%s not existing." % tile_features_test_dir)
        exit()
    if not os.path.exists(gene_features_train_dir):
        print("%s not existing." % gene_features_train_dir)
        exit()
    if not os.path.exists(gene_features_test_dir):
        print("%s not existing." % gene_features_test_dir)
        exit()

    # get features:
    tile_features_train = get_train_data(tile_features_train_dir)
    tile_features_test = get_train_data(tile_features_test_dir)
    gene_features_train = get_train_data(gene_features_train_dir)
    gene_features_test = get_train_data(gene_features_test_dir)

    # concatenation of tile and gene features:
    if args.method == 'fine_tuning':
        gene_copy_ratio = 20 # TODO vedere se 20 va bene (tiles dim: 2048, genes dim: 100->100*20=2000)
    else:
        gene_copy_ratio = 1

    X_train = feature_concatenation.concatenate(tile_features_train, gene_features_train, gene_copy_ratio)
    X_test = feature_concatenation.concatenate(tile_features_test, gene_features_test, gene_copy_ratio)

    # balance the training dataset
    if args.balancing:
        X_train, y_train = class_balancing.balance_dataset(X_train, y_train, args.balancing)

    if args.method == 'fixed_feature_generator':
        # TODO classify data with SVM
        pass
    elif args.method == 'fine_tuning':
        # TODO classify data with NN (?)
        pass

'''
    normal_labels_list = np.zeros((len(normal_tile_features),), dtype=int)
    tumor_labels_list = np.ones((len(tumor_tile_features),), dtype=int)
    
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

    # vgg16:
    # model = keras.Sequential([
    #     keras.layers.Dense(4096, activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(4096, activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dense(2, activation='softmax')
    # ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_features, train_labels, epochs=10, shuffle=True, validation_split=0.5)
'''

if __name__ == '__main__':
    main()