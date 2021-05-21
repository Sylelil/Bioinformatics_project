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
    n_comp = config.get('general', 'num_principal_components')
    if n_comp.isdecimal():
        params['general']['num_principal_components'] = int(n_comp)
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

    return params

