import configparser
import pandas as pd
import numpy as np
from pathlib import Path
from config import paths
from sklearn.model_selection import GroupKFold


def get_patient_kfold_split(X_train, y_train, data_info_path, n_splits):

    # data info path dovrebbero essere questi:
    # - nel caso di PCA è "pca200/info_train.csv"
    # - altrimenti "all/info_train.csv"
    
    # n_splits si prende da param (attenzione, si prende in modi diversi in shallow_classifier e in nn_classifier)

    train_info_df = pd.read_csv(data_info_path)
    X_train_filenames = train_info_df['filename']  # per ogni sample, ricavo il filename

    train_filenames_file = Path(paths.filename_splits_dir) / 'train_filenames.npy'
    train_filenames = np.load(train_filenames_file)         # lista di tutti i filename nel train dataset (senza duplicati)

    group_nums_range = np.arange(len(train_filenames))      # [0,1,2,3,4, ..., len(train_filenames)-1]
                                                            # -> un numero per ciascun gruppo (gruppo = filename)

    train_filenames_group_nums = {}  # {filename0:0, filename1:1, filename2:2, ...}
    for filename, group_num in zip(train_filenames, group_nums_range):
        train_filenames_group_nums[filename] = group_num

    groups = []     # parallelo a X_train: per ogni sample di X_train, in groups segno il suo numero di gruppo
    for sample_filename in X_train_filenames:
        try:
            groups.append(train_filenames_group_nums[sample_filename])
        except KeyError:
            groups.append(0) # Inseriamo tutte le tile di cui non troviamo il case id in maniera consistente nello stesso gruppo

    group_kfold = GroupKFold(n_splits=n_splits)

    return group_kfold.split(X_train, y_train, groups), groups


def get_concatenated_data_old(data_path):
    """
       Description: Apply StandardScaler and IncrementalPCA to data.
       :param data_path: path of data.
       :returns: X_train, y_train, X_val, y_val, X_test, y_test: data and labels
    """
    print('>> Reading data files...')
    X_train = pd.read_csv(Path(data_path) / 'x_train.csv', delimiter=',', header=None).values
    y_train = pd.read_csv(Path(data_path) / 'y_train.csv', delimiter=',', header=None).values
    X_val = pd.read_csv(Path(data_path) / 'x_val.csv', delimiter=',', header=None).values
    y_val = pd.read_csv(Path(data_path) / 'y_val.csv', delimiter=',', header=None).values
    X_test = pd.read_csv(Path(data_path) / 'x_test.csv', delimiter=',', header=None).values
    y_test = pd.read_csv(Path(data_path) / 'y_test.csv', delimiter=',', header=None).values

    return X_train, y_train.ravel().astype(int), X_val, y_val.ravel().astype(int), X_test, y_test.ravel().astype(int)


def get_data(data_path):
    """
       Description: Get data for classification.
       :param data_path: path of data.
       :returns: X_train, y_train, X_test, y_test: data and labels
    """
    print('>> Reading train files...')
    X_train = pd.read_csv(Path(data_path) / 'x_train.csv', delimiter=',', header=None).values
    y_train = pd.read_csv(Path(data_path) / 'y_train.csv', delimiter=',', header=None).values
    print('>> Reading test files...')
    X_test = pd.read_csv(Path(data_path) / 'x_test.csv', delimiter=',', header=None).values
    y_test = pd.read_csv(Path(data_path) / 'y_test.csv', delimiter=',', header=None).values

    return X_train, y_train.ravel().astype(int), X_test, y_test.ravel().astype(int)


def read_config_file(config_file_path):
    """
       Description: Read configuration parameters.
       :param config_file_path: Path of the configuration file.
       :returns: Dictionary of parameters
    """
    params = {}
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file_path)

    # general
    params['general'] = {}
    params['general']['random_state'] = config.getint('general', 'random_state')
    params['general']['use_features_images_only'] = config.getboolean('general', 'use_features_images_only')
    params['general']['apply_pca_to_features_images'] = config.getboolean('general', 'apply_pca_to_features_images')
    if params['general']['apply_pca_to_features_images']:
        n_comp = config.get('general', 'num_principal_components')
        if n_comp.isdecimal():
            params['general']['num_principal_components'] = int(n_comp)
        else:
            print(f'error: invalid <num_principal_components> in src/config/integration/conf.ini: {n_comp}')
    else:
        params['general']['num_principal_components'] = None


    # cv
    params['cv'] = {}
    params['cv']['n_outer_splits'] = config.getint('cv', 'n_outer_splits')
    params['cv']['n_inner_splits'] = config.getint('cv', 'n_inner_splits')

    # nn
    params['nn'] = {}
    params['nn']['epochs'] = config.getint('nn', 'epochs')
    params['nn']['batchsize'] = config.getint('nn', 'batchsize')
    params['nn']['units_1'] = config.getint('nn', 'units_1')
    params['nn']['units_2'] = config.getint('nn', 'units_2')
    params['nn']['lr'] = config.getfloat('nn', 'lr')
    params['nn']['n_inner_splits'] = config.getint('nn', 'n_inner_splits')
    params['nn']['random_state'] = config.getint('nn', 'random_state')

    # pca_nn
    params['pcann'] = {}
    params['pcann']['epochs'] = config.getint('pcann', 'epochs')
    params['pcann']['batchsize'] = config.getint('pcann', 'batchsize')
    params['pcann']['units_1'] = config.getint('pcann', 'units_1')
    params['pcann']['units_2'] = config.getint('pcann', 'units_2')
    params['pcann']['lr'] = config.getfloat('pcann', 'lr')
    params['pcann']['random_state'] = config.getint('pcann', 'random_state')
    params['pcann']['n_inner_splits'] = config.getint('pcann', 'n_inner_splits')

    # linearsvc
    params['linearsvc'] = {}
    params['linearsvc']['max_iter'] = config.getint('linearsvc', 'max_iter')

    # sgd
    params['sgdclassifier'] = {}
    params['sgdclassifier']['max_iter'] = config.getint('sgdclassifier', 'max_iter')

    return params

