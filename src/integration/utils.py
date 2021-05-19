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


def get_concatenated_data(data_path, n_features_images=None):
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

    if n_features_images:
        n = n_features_images
        X_train = X_train[:, 0:n]
        X_val = X_val[:, 0:n]
        X_test = X_test[:, 0:n]

    return X_train, y_train.ravel().astype(int), X_val, y_val.ravel().astype(int), X_test, y_test.ravel().astype(int)


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
    params['general']['use_pca_scaled_features'] = config.getboolean('general', 'use_pca_scaled_features')
    n_features_images = config['general']['n_features_images']
    if n_features_images == 'None' or n_features_images == '':
        params['general']['n_features_images'] = None
    else:
        params['general']['n_features_images'] = config.getint('general', 'n_features_images')

    # preprocessing
    params['preprocessing'] = {}
    params['preprocessing']['batchsize'] = config.getint('preprocessing', 'batchsize')

    # nn
    params['nn'] = {}
    params['nn']['epochs'] = config.getint('nn', 'epochs')
    params['nn']['batchsize'] = config.getint('nn', 'batchsize')
    params['nn']['units_1'] = config.getint('nn', 'units_1')
    params['nn']['units_2'] = config.getint('nn', 'units_2')
    params['nn']['lr'] = config.getfloat('nn', 'lr')

    # pca_nn
    params['pcann'] = {}
    params['pcann']['epochs'] = config.getint('pcann', 'epochs')
    params['pcann']['batchsize'] = config.getint('pcann', 'batchsize')
    params['pcann']['units_1'] = config.getint('pcann', 'units_1')
    params['pcann']['units_2'] = config.getint('pcann', 'units_2')
    params['pcann']['lr'] = config.getfloat('pcann', 'lr')

    # linearsvc
    params['linearsvc'] = {}
    params['linearsvc']['max_iter'] = config.getint('linearsvc', 'max_iter')

    # sgd
    params['sgdclassifier'] = {}
    params['sgdclassifier']['max_iter'] = config.getint('sgdclassifier', 'max_iter')

    # pca
    params['pca'] = {}
    params['pca']['n_components'] = None

    return params

