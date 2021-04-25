import configparser
import os
import sys
from sklearn import metrics
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def get_tile_features(lookup_dir):
    """
       Description: Private function. Read extracted tile features from files.
       :param lookup_dir: Path of the lookup directory.
       :return: Dataframe of features.
    """
    all_tiles_features = []  # list of all tile features
    slide_data = None

    for np_file in tqdm(os.listdir(lookup_dir)):
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
        caseid_label_col = [[caseid, label]] * slide_data.shape[0]
        slide_data_caseid_label = np.append(slide_data, caseid_label_col, axis=1)
        # shape of slide_data_caseid_label:
        #   [[coordx_1,coordy_1, feat1_1, feat2_1, feat3_1, ..., caseid, label]   # tile 1
        #    [coordx_2,coordy_2, feat1_2, feat2_2, feat3_2, ..., caseid, label]   # tile 2
        #    [coordx_3,coordy_3, feat1_3, feat2_3, feat3_3, ..., caseid, label]   # tile 3
        #    [...]]                                                               # tile ...

        # add to list of all tile features
        slide_data_list = list(slide_data_caseid_label)
        all_tiles_features.extend(slide_data_list)

    print(f'list shape: ({len(all_tiles_features)}, {len(all_tiles_features[0])})')

    # convert to dataframe
    col_names = ['coord0', 'coord1']
    col_names.extend([f'feat_t_{x}' for x in range(slide_data.shape[1] - 2)])
    col_names.extend(['caseid', 'label'])
    print(">> Converting to dataframe...")
    df_tiles_features = pd.DataFrame(all_tiles_features, columns=col_names).set_index('caseid')

    print(f'shape: {df_tiles_features.shape}')

    return df_tiles_features


def get_gene_features(lookup_dir):
    """
       Description: Private function. Read extracted gene features from files.
       :param lookup_dir: Path of the lookup directory.
       :return: Dataframe of features.
    """
    all_features = []  # list of all tile features
    data = None

    for np_file in tqdm(os.listdir(lookup_dir)):
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
        all_features.append(data_list)

    # convert to dataframe
    col_names = [f'feat_g_{x}' for x in range(data.shape[0])]
    col_names.extend(['caseid', 'label'])
    df_gene_features = pd.DataFrame(all_features, columns=col_names).set_index('caseid')

    print(f'shape: {df_gene_features.shape}')

    return df_gene_features


def read_extracted_features():
    """
       Description: Read extracted features from results folders.
       :return: Train and test splits of gene and tile data.
    """

    tile_features_train_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'images' / 'training'
    tile_features_test_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'images' / 'test'
    gene_features_train_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'genes' / 'training'
    gene_features_test_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'genes' / 'test'

    if not os.path.exists(Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results'):
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
    print(">> Reading gene features for training ...")
    gene_features_train = get_gene_features(gene_features_train_dir)
    print(">> Reading gene features for training ...")
    gene_features_test = get_gene_features(gene_features_test_dir)
    print(">> Reading tile features for testing ...")
    tile_features_test = get_tile_features(tile_features_test_dir)
    print(">> Reading tile features for training ...")
    tile_features_train = get_tile_features(tile_features_train_dir)

    print('>> Done')

    return tile_features_train, tile_features_test, gene_features_train, gene_features_test


def read_config_file(config_file_path):
    """
       Description: Read configuration parameters.
       :param config_file_path: Path of the configuration file.
       :param method: Classification method.
       :return: Dictionary of parameters
    """
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # general
    scoring = config['general']['scoring']
    if scoring == 'matthews_corrcoef':
        params['scoring'] = metrics.matthews_corrcoef
    if scoring == 'accuracy':
        params['scoring'] = metrics.accuracy_score

    random_state = config['general']['random_state']
    if random_state == 'None' or random_state == '':
        params['random_state'] = None
    else:
        params['random_state'] = config.getint('general', 'random_state')

    params['batchsize'] = config.getint('general', 'batchsize')

    params['cv_inner_n_splits'] = config.getint('crossvalidation', 'cv_inner_n_splits')
    params['cv_outer_n_splits'] = config.getint('crossvalidation', 'cv_outer_n_splits')
    params['cv_n_splits'] = config.getint('crossvalidation', 'cv_n_splits')

    # pca
    params['percentage_of_variance'] = config.getfloat('pca', 'percentage_of_variance')
    params['n_components'] = config.getint('pca', 'n_components')

    #nn
    params['epochs'] = config.getint('nn', 'epochs')
    params['batchsize_nn'] = config.getint('nn', 'batchsize')


    return params
