import configparser
from sklearn import metrics
import numpy as np


def compute_class_weights(labels):
    """
        Description: Compute class weights from list of labels.
        :param labels: list of labels.
        :returns: class weights dictionary
    """
    print(">> Computing class weights...")
    total = len(labels)
    pos = np.count_nonzero(labels)
    neg = total - pos
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * total / 2.0
    weight_for_1 = (1 / pos) * total / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight


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

    # pca
    params['pca'] = {}
    params['pca']['percentage_of_variance'] = config.getfloat('pca', 'percentage_of_variance')
    params['pca']['n_components'] = config.getint('pca', 'n_components')

    # preprocessing
    params['preprocessing'] = {}
    params['preprocessing']['batchsize'] = config.getint('preprocessing', 'batchsize')

    # nn
    params['nn'] = {}
    params['nn']['epochs'] = config.getint('nn', 'epochs')
    params['nn']['batchsize'] = config.getint('nn', 'batchsize')

    return params
