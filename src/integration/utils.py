import configparser
import os

from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

from config import paths
from src.common import plots

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def __scaling_pca(params, concatenated_pca_path, train_filepath, val_filepath, test_filepath,
                  get_explained_variance_ratio=False):
    x_train_pca_path = Path(concatenated_pca_path) / 'x_train.npy'
    y_train_pca_path = Path(concatenated_pca_path) / 'y_train.npy'
    x_val_pca_path = Path(concatenated_pca_path) / 'x_val.npy'
    y_val_pca_path = Path(concatenated_pca_path) / 'y_val.npy'
    x_test_pca_path = Path(concatenated_pca_path) / 'x_test.npy'
    y_test_pca_path = Path(concatenated_pca_path) / 'y_test.npy'

    batchsize = params['preprocessing']['batchsize']

    print(">> Fitting scaler on train data...")
    scaler = StandardScaler()
    for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_train_chunk = chunk.iloc[:, :-1]
        scaler.partial_fit(X_train_chunk)

    ipca = IncrementalPCA(n_components=params['pca']['n_components'])
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    print(
        f">> Transforming train data with scaler and fitting incremental pca ({params['pca']['n_components']} components)...")
    for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_train_chunk = chunk.iloc[:, :-1]
        X_train_chunk_scaled = scaler.transform(X_train_chunk)
        ipca.partial_fit(X_train_chunk_scaled)
    print(">> Transforming train data with incremental pca...")
    for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_train_chunk = chunk.iloc[:, :-1]
        y_train_chunk = chunk['label']
        X_train_chunk_scaled = scaler.transform(X_train_chunk)
        X_train_chunk_ipca = ipca.transform(X_train_chunk_scaled)
        X_train.extend(X_train_chunk_ipca)
        y_train.extend(y_train_chunk)
    print(">> Transforming validation data with incremental pca...")
    for chunk in tqdm(pd.read_csv(val_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_val_chunk = chunk.iloc[:, :-1]
        y_val_chunk = chunk['label']
        X_val_chunk_scaled = scaler.transform(X_val_chunk)
        X_val_chunk_ipca = ipca.transform(X_val_chunk_scaled)
        X_val.extend(X_val_chunk_ipca)
        y_val.extend(y_val_chunk)
    print(">> Transforming test data with incremental pca...")
    for chunk in tqdm(pd.read_csv(test_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_test_chunk = chunk.iloc[:, :-1]
        y_test_chunk = chunk['label']
        X_test_chunk_scaled = scaler.transform(X_test_chunk)
        X_test_chunk_ipca = ipca.transform(X_test_chunk_scaled)
        X_test.extend(X_test_chunk_ipca)
        y_test.extend(y_test_chunk)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    print(">> Saving computed features on files in assets/concatenated_pca/ folder...")
    np.save(x_train_pca_path, X_train)
    np.save(y_train_pca_path, y_train)
    np.save(x_val_pca_path, X_val)
    np.save(y_val_pca_path, y_val)
    np.save(x_test_pca_path, X_test)
    np.save(y_test_pca_path, y_test)

    if get_explained_variance_ratio:
        return ipca.explained_variance_ratio_

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_concatenated_data(data_path):
    """
       Description: Apply StandardScaler and IncrementalPCA to data.
       :param train_filepath: path of train data.
       :param val_filepath: path of validation data.
       :param test_filepath: path of test data.
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


def compute_scaling_pca(params, train_filepath, val_filepath, test_filepath):
    """
       Description: Apply StandardScaler and IncrementalPCA to data.
       :param params: configuration parameters.
       :param train_filepath: path of train data.
       :param val_filepath: path of validation data.
       :param test_filepath: path of test data.
       :returns: X_train, y_train, X_val, y_val, X_test, y_test: data and labels
    """
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    X_test = None
    y_test = None
    if params['general']['use_pca_scaled_features']:
        concatenated_pca_path = paths.concatenated_results_dir / f"pca_{params['pca']['n_components']}"
    else:
        concatenated_pca_path = paths.concatenated_pca_dir / f"concatenated_pca_{params['pca']['n_components']}"

    if os.path.exists(concatenated_pca_path):
        print('>> Reading files with scaled and pca data previously computed...')
        X_train = np.load(Path(concatenated_pca_path) / 'x_train.npy')
        y_train = np.load(Path(concatenated_pca_path) / 'y_train.npy')
        X_val = np.load(Path(concatenated_pca_path) / 'x_val.npy')
        y_val = np.load(Path(concatenated_pca_path) / 'y_val.npy')
        X_test = np.load(Path(concatenated_pca_path) / 'x_test.npy')
        y_test = np.load(Path(concatenated_pca_path) / 'y_test.npy')
    else:
        if params['general']['use_pca_scaled_features']:
            print(f'error: path does not exist: {concatenated_pca_path}.')
            exit()
        else:
            os.mkdir(concatenated_pca_path)
            X_train, y_train, X_val, y_val, X_test, y_test = __scaling_pca(params, concatenated_pca_path, train_filepath,
                                                                           val_filepath, test_filepath)

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_explained_variance_pca(params, train_filepath, val_filepath, test_filepath):
    print(
        f">> Starting procedure to plot explained variance of Principal Components (up to {params['pca']['n_components']} components)...")
    concatenated_pca_path = paths.concatenated_pca_dir / f"concatenated_pca_{params['pca']['n_components']}"
    if not os.path.exists(concatenated_pca_path):
        os.makedirs(concatenated_pca_path)
    explained_variance_ratio = __scaling_pca(params, concatenated_pca_path, train_filepath,
                                                                 val_filepath, test_filepath,
                                                                 get_explained_variance_ratio=True)
    plots.plot_explained_variance(explained_variance_ratio, concatenated_pca_path)
    print('>> Done.')


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
    random_state = config['general']['random_state']
    if random_state == 'None' or random_state == '':
        params['general']['random_state'] = None
    else:
        params['general']['random_state'] = config.getint('general', 'random_state')
    params['general']['use_pca_scaled_features'] = config.getboolean('general', 'use_pca_scaled_features')

    # preprocessing
    params['preprocessing'] = {}
    params['preprocessing']['batchsize'] = config.getint('preprocessing', 'batchsize')

    params['linearsvc'] = {}
    params['linearsvc']['max_iter'] = config.getint('linearsvc', 'max_iter')

    params['sgd'] = {}
    params['sgd']['max_iter'] = config.getint('sgd', 'max_iter')

    # nn
    params['nn'] = {}
    params['nn']['epochs'] = config.getint('nn', 'epochs')
    params['nn']['batchsize'] = config.getint('nn', 'batchsize')
    params['nn']['units_1'] = config.getint('nn', 'units_1')
    params['nn']['units_2'] = config.getint('nn', 'units_2')

    # pca
    params['pca'] = {}
    params['pca']['n_components'] = None

    return params

