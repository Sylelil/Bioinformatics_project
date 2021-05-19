from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import metrics
from tensorflow import keras
from config import paths
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import ClusterCentroids
import tensorflow.keras.backend as K
import tensorflow_addons as tfa


def get_balancing_method(method, params):
    """
       Description: Return method corresponding to the parameter 'method'.
       :param method: Class balancing method.
       :param params: Parameters from configuration file.
       :returns: Class balancing method.
    """
    if method == 'randomupsampling':
        return RandomOverSampler(random_state=params['general']['random_state'])
    elif method == 'smote':
        return SMOTE(random_state=params['general']['random_state'])
    return None


def compute_class_weights(labels):
    """
        Description: Compute class weights from list of labels.
        :param labels: list of labels.
        :returns: class weights dictionary.
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
