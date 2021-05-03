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


''' Metrics for shallow classification: '''
METRICS_skl = [
    metrics.accuracy_score,
    metrics.matthews_corrcoef,
    metrics.average_precision_score,
    metrics.f1_score,
    metrics.precision_score,
    metrics.recall_score,
    metrics.roc_auc_score,
]


def matthews_correlation(y_true, y_pred):
    """
       Description: Compute Matthews Correlation Coefficient.
       :param y_true: ground truth labels.
       :param y_pred: predicted labels.
       :returns: Matthews Correlation Coefficient.
    """

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


''' Metrics for nn classification: '''
METRICS_keras = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    matthews_correlation,
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def get_balancing_method(method, params):
    """
       Description: Return method corresponding to the parameter 'method'.
       :param method: Class balancing method.
       :param params: Parameters from configuration file.
       :returns: Class balancing method.
    """
    if method == 'random_upsampling':
        return RandomOverSampler(random_state=params['general']['random_state'])
    elif method == 'combined':
        return SMOTEENN(random_state=params['general']['random_state'])
    elif method == 'smote':
        return SMOTE(random_state=params['general']['random_state'])
    elif method == 'downsampling':
        return ClusterCentroids(random_state=params['general']['random_state'])
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
