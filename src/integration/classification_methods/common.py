import os
import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn import metrics
from tensorflow import keras
from config import paths
from src.integration import plots
import matplotlib.pyplot as plt

METRICS_skl = [
    metrics.accuracy_score,
    metrics.average_precision_score,
    metrics.f1_score,
    metrics.precision_score,
    metrics.recall_score,
    metrics.matthews_corrcoef,
    metrics.roc_auc_score,
    metrics.confusion_matrix,
]


METRICS_keras = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


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


def compute_scaling_pca(params, train_filepath, val_filepath, test_filepath):
    """
       Description: Apply StandardScaler and IncrementalPCA to data.
       :param params: configuration parameters.
       :param train_filepath: path of train data.
       :param val_filepath: path of validation data.
       :param test_filepath: path of test data.
       :returns: X_train, y_train, X_val, y_val, X_test, y_test: data and labels
    """
    x_train_pca_path = Path('assets') / 'concatenated_pca' / 'x_train.npy'
    y_train_pca_path = Path('assets') / 'concatenated_pca' / 'y_train.npy'
    x_val_pca_path = Path('assets') / 'concatenated_pca' / 'x_val.npy'
    y_val_pca_path = Path('assets') / 'concatenated_pca' / 'y_val.npy'
    x_test_pca_path = Path('assets') / 'concatenated_pca' / 'x_test.npy'
    y_test_pca_path = Path('assets') / 'concatenated_pca' / 'y_test.npy'

    if os.path.exists(Path('assets') / 'concatenated_pca'):
        print('>> Reading files with scaled and pca data previously computed...')
        X_train = np.load(x_train_pca_path)
        y_train = np.load(y_train_pca_path)
        X_val = np.load(x_val_pca_path)
        y_val = np.load(y_val_pca_path)
        X_test = np.load(x_test_pca_path)
        y_test = np.load(y_test_pca_path)

    else:
        os.mkdir(Path('assets') / 'concatenated_pca')
        batchsize = params['preprocessing']['batchsize']

        print(">> Fitting scaler...")
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
        print(">> Transforming train data with scaler and fitting incremental pca...")
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

    return X_train, y_train, X_val, y_val, X_test, y_test


def aggregate_results_per_patient(y_pred_test):
    """
        Description: Aggregate results for each patient in test dataset.
        :param y_pred_test: list of predictions.
    """
    print('>> Aggregating results for each patient in test dataset...')
    test_data_info_path = Path(paths.concatenated_results_dir) / 'test' / 'concat_data_info.csv'
    test_info_df = pd.read_csv(test_data_info_path)
    test_info_df['y_pred_test'] = y_pred_test
    # test_filenames_file = Path(paths.filename_splits_dir) / 'test_filenames.npy'
    test_filenames_file = Path(paths.filename_splits_dir) / 'test_caseids.npy'
    test_filenames = np.load(test_filenames_file)
    gt_labels = []
    pred_labels = []
    filename_list = []
    '''
    for filename in tqdm(test_filenames):
        if filename not in filename_list:
            patient_info = test_info_df.loc[test_info_df['filename'] == filename]
            patient_predictions = list(patient_info['y_pred_test'])
            gt_label = filename[-1]
            pred_label = 1 if 1 in patient_predictions else 0  # if at least one prediction is tumor, the predicted label is tumor
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)
        else:  # filename of patient already aggregated
            continue
    '''
    for filename in tqdm(test_filenames):
        if filename not in filename_list:
            options = [f"{filename}_0", f"{filename}_1"]
            # patient_info = test_info_df.loc[test_info_df['filename'] == filename]
            patient_info = test_info_df.loc[test_info_df['filename'].isin(options)]
            patient_predictions = list(patient_info['y_pred_test'])
            gt_label = list(patient_info['label'])[0]
            #gt_label = filename[-1]
            pred_label = 1 if 1 in patient_predictions else 0  # if at least one prediction is tumor, the predicted label is tumor
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)
        else:  # filename of patient already aggregated
            continue

    print('Test scores for results aggregated by patient:')
    test_scores_aggregated = []
    for metric in METRICS_skl:
        test_scores_aggregated.append((metric.__name__, metric(gt_labels, pred_labels)))
    for name, value in test_scores_aggregated:
        print(name, ': ', value)
    print()

    plots.plot_test_results_aggregated(gt_labels, np.asarray(pred_labels))
    plt.show()

