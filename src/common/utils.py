import configparser
import os
import sys
from sklearn import metrics
import numpy as np
import pandas as pd
from pathlib import Path


def read_config_file(config_file_path, method):
    """
       Description: Read configuration parameters.
       :param config_file_path: Path of the configuration file.
       :param method: Classification method.
       :return: Dictionary of parameters
    """
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file_path)

    scoring = config['general']['scoring']
    if scoring == 'matthews_corrcoef':
        params['scoring'] = metrics.matthews_corrcoef

    random_state = config['general']['random_state']
    if random_state == 'None' or random_state == '':
        params['random_state'] = None
    else:
        params['random_state'] = config.getint('general', 'random_state')

    if method == 'svm':
        params['percentage_of_variance'] = config.getfloat('pca', 'percentage_of_variance')
        params['cv_grid_search_rank'] = config.getint('svm', 'cv_grid_search_rank')
        if config['svm']['kernel'] == 'linear' or config['svm']['kernel'] == 'rbf':
            params['kernel'] = config['svm']['kernel']
        else:
            sys.stderr.write("Invalid value for <kernel> in config file")
            exit(1)
    elif method == 'nn':
        pass  # TODO
    elif method == 'pca_nn':
        params['percentage_of_variance'] = config.getfloat('pca', 'percentage_of_variance')
        pass  # TODO
    else:
        sys.stderr.write("Invalid value for <classification method> in config file")
        exit(1)

    return params
