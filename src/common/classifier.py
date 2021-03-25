import argparse
import os
import sys
#sys.path.insert(1, 'c:/Users/rosee/workspace_Polito/git/Bioinformatics_project/src/common/')
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("PATH:", os.environ.get('PATH'))
from os import path
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from src.common import class_balancing, feature_concatenation
import class_balancing, feature_concatenation


'''
def get_data(lookup_dir):
    features_list = []

    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        data = np.load(file_path)
        print(data.shape)
        np_features_list = list(data)
        features_list.extend(np_features_list) # TODO

    return np.array(features_list)
'''

def get_tile_data(lookup_dir):
    all_tiles_features = [] # list of all tile features

    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        filename = os.path.splitext(np_file)[0]
        caseid = filename[:-2]
        label = filename[-1]

        # get list of tile features of a single slide from file
        slide_data = np.load(file_path) 
        # shape of slide_data:
        #   [[coordx_1,coordy_1, feat1_1, feat2_1, feat3_1, ...]   # tile 1
        #    [coordx_2,coordy_2, feat1_2, feat2_2, feat3_2, ...]   # tile 2
        #    [coordx_3,coordy_3, feat1_3, feat2_3, feat3_3, ...]   # tile 3
        #    [...]]                                                # tile ...
        

        # append to each row of dataframe the caseid and label of that slide
        caseid_label_col = [[caseid, label]]*slide_data.shape[0]
        slide_data_caseid_label = np.append(slide_data, caseid_label_col, axis=1)
        # shape of slide_data_caseid_label:
        #   [[coordx_1,coordy_1, feat1_1, feat2_1, feat3_1, ..., caseid, label]   # tile 1
        #    [coordx_2,coordy_2, feat1_2, feat2_2, feat3_2, ..., caseid, label]   # tile 2
        #    [coordx_3,coordy_3, feat1_3, feat2_3, feat3_3, ..., caseid, label]   # tile 3
        #    [...]]                                                               # tile ...

        # add to list of all tile features
        slide_data_list = list(slide_data_caseid_label)
        all_tiles_features.extend(slide_data_list)
        
    # convert to dataframe
    col_names = ['coord0','coord1']
    col_names.extend([f'feat{x}' for x in range(slide_data.shape[1] - 2)])
    col_names.extend(['caseid','label'])
    df_tiles_features = pd.DataFrame(all_tiles_features, columns=col_names)
    
    print(df_tiles_features.shape)

    return df_tiles_features


def get_gene_data(lookup_dir):
    all_features = [] # list of all tile features

    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        filename = os.path.splitext(np_file)[0]
        caseid = filename[:-2]
        label = filename[-1]

        # get features of a patient from file
        data = np.load(file_path) 
        # shape of data:
        #   [feat1, feat2, feat3, ...] 
        

        # append to each row of dataframe the caseid and label of that slide
        data_caseid_label = np.append(data, [caseid, label])
        # shape of data_caseid_label:
        #   [feat1, feat2, feat3, ..., caseid, label]

        # add to list of all tile features
        data_list = list(data_caseid_label)
        all_features.extend(data_list)
        
    # convert to dataframe
    col_names = [f'feat{x}' for x in range(data.shape[0])]
    col_names.extend(['caseid','label'])
    df_gene_features = pd.DataFrame(all_features, columns=col_names)

    print(df_gene_features.shape)

    return df_gene_features


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

    # Read features from file
    print(">> Reading features from files...")
    
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
    tile_features_train = get_tile_data(tile_features_train_dir)
    tile_features_test = get_tile_data(tile_features_test_dir)
    gene_features_train = get_gene_data(gene_features_train_dir)
    gene_features_test = get_gene_data(gene_features_test_dir)

    print(f">> tile_features_train: {tile_features_train.shape}")
    print(f">> tile_features_test: {tile_features_test.shape}")
    print(f">> gene_features_train: {gene_features_train.shape}")
    print(f">> gene_features_test: {gene_features_test.shape}")



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