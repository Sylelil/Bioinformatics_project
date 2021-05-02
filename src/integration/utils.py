import configparser
import os
import sys

from pandas import DataFrame
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

from config import paths
from src.integration import plots
from src.integration.classification_methods import common

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


def compute_scaling_pca(params, train_filepath, val_filepath, test_filepath):
    """
       Description: Apply StandardScaler and IncrementalPCA to data.
       :param params: configuration parameters.
       :param train_filepath: path of train data.
       :param val_filepath: path of validation data.
       :param test_filepath: path of test data.
       :returns: X_train, y_train, X_val, y_val, X_test, y_test: data and labels
    """
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
    print(">> Plotting individual and cumulative explained variance...")
    plt.figure()
    plt.plot(np.cumsum(explained_variance_ratio), label='Cumulative explained variance', linewidth=2, marker='.')
    plt.plot(explained_variance_ratio, label='Individual explained variance', linewidth=2, marker='.')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Number of components')
    plt.legend(loc='best')
    plt.savefig(Path(concatenated_pca_path) / 'explained_variance')
    plt.show()
    print('>> Done.')


def __classification_report(y_test, y_pred_test, test_scores):
    report = {}

    # compute confusion matrix values:
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    report['Normal Detected (True Negatives)'] = confusion_matrix[0][0]
    report['Normal Incorrectly Detected (False Positives)'] = confusion_matrix[0][1]
    report['Tumor Missed (False Negatives)'] = confusion_matrix[1][0]
    report['Tumor Detected (True Positives)'] = confusion_matrix[1][1]
    report['Total Tumor'] = np.sum(confusion_matrix[1])

    # add test scores:
    report.update(test_scores)

    # compute patch score and patient score:
    report['patch_score'] = common.compute_patch_score(y_test, y_pred_test)
    report['patient_avg_score'], patent_stddev_score = common.compute_patient_score(y_pred_test)

    # compute further per-class scores:
    per_class_report_dict = metrics.classification_report(y_test, y_pred_test)

    return report, per_class_report_dict


def __print_classification_report(experiment_info, report, per_class_report, file=sys.stdout):
    print('Experiment details:', file=file)
    print('-------------------', file=file)
    for k, v in experiment_info.items():
        print("{:<50} {:<15}".format(k, v), file=file)
    print(file=file)

    print('Test results:', file=file)
    print('-------------', file=file)
    for k, v in report.items():
        print("{:<50} {:<15}".format(k, v), file=file)
    print(file=file)
    print(per_class_report, file=file)


def generate_classification_report(save_path, y_test, y_pred_test, test_scores, experiment_info):
    # generate report:
    print('>> Generating classification report...')
    report, per_class_report = __classification_report(y_test, y_pred_test, test_scores)

    # print on stdout:
    __print_classification_report(experiment_info, report, per_class_report)

    # print on file:
    report_path = Path(save_path) / 'report.txt'
    print(f'>> Saving classification report on file {report_path}...')
    with open(report_path, 'w') as f:
        __print_classification_report(experiment_info, report, per_class_report, file=f)

'''
def generate_classification_plots(save_path, classifier, X_train, y_train, X_test, y_test):
    print(f'Generating classification plots, saved in {save_path}...')

    # plot confusion matrix
    plt.figure()
    metrics.plot_confusion_matrix(classifier, X_test, y_test, values_format='')
    plt.savefig(Path(save_path) / 'confusion_matrix')

    # plot roc
    plt.figure()
    metrics.plot_roc_curve(classifier, X_train, y_train, ax=plt.gca(), name="Train")
    metrics.plot_roc_curve(classifier, X_test, y_test, ax=plt.gca(), name="Test")
    plt.legend(loc='lower right')
    plt.savefig(Path(save_path) / 'roc')

    # plot precision-recall curve
    plt.figure()
    metrics.plot_precision_recall_curve(classifier, X_train, y_train, ax=plt.gca(), name="Train")
    metrics.plot_precision_recall_curve(classifier, X_test, y_test, ax=plt.gca(), name="Test")
    plt.legend(loc='lower right')
    plt.savefig(Path(save_path) / 'prc')

    plt.show()
'''

def generate_classification_plots(save_path, y_test, y_pred_test, y_train, y_pred_train):
    print(f'>> Generating classification plots and saving them in {save_path}...')

    # plot confusion matrix
    plt.figure()
    plots.plot_cm(y_test, y_pred_test)
    plt.savefig(Path(save_path) / 'confusion_matrix')

    # plot roc
    plt.figure()
    plots.plot_roc("Train", y_train, y_pred_train)
    plots.plot_roc("Test", y_test, y_pred_test)
    plt.legend(loc='lower right')
    plt.savefig(Path(save_path) / 'roc')

    # plot precision-recall curve
    plt.figure()
    plots.plot_prc("Train", y_train, y_pred_train)
    plots.plot_prc("Test", y_test, y_pred_test)
    plt.legend(loc='lower right')
    plt.savefig(Path(save_path) / 'prc')

    plt.show()


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

    # preprocessing
    params['preprocessing'] = {}
    params['preprocessing']['batchsize'] = config.getint('preprocessing', 'batchsize')

    # nn
    params['nn'] = {}
    params['nn']['epochs'] = config.getint('nn', 'epochs')
    params['nn']['batchsize'] = config.getint('nn', 'batchsize')

    # pca
    params['pca'] = {}
    params['pca']['n_components'] = None

    return params

