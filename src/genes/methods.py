import configparser
import os
import sys
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from scipy import stats


def read_gene_expression_data(path):
    data_frame_0 = pd.DataFrame()
    data_frame_1 = pd.DataFrame()

    for file_name in tqdm(os.listdir(path), desc=">> Reading patient data...", file=sys.stdout):
        file_path = os.path.join(path, file_name)
        with open(file_path) as f:
            patient_df = pd.read_csv(f, sep="\t", header=None, index_col=0, names=[file_name.replace(".txt", "")])
            patient_df = pd.DataFrame.transpose(patient_df)
            if file_name.endswith("_0.txt"):
                data_frame_0 = data_frame_0.append(patient_df)
            else:
                data_frame_1 = data_frame_1.append(patient_df)
    df_patients = data_frame_0.append(data_frame_1, sort=False)  # Merge normal data frame with tumor data frame
    return df_patients


def load_selected_genes(selected_features_dir):
    X = []
    y = []
    for patient_file in tqdm(os.listdir(selected_features_dir), desc=">> Reading selected genes...", file=sys.stdout):
        patient_features = np.load(os.path.join(selected_features_dir, patient_file))
        case_id = os.path.splitext(patient_file)[0]
        target = case_id[-1:]
        X.append(patient_features)
        y.append(int(target))

    return X, y


def save_selected_genes(X_train, X_test, selected_genes, results_dir, selected_genes_path):

    fp = open(selected_genes_path, "w")
    for gene in selected_genes:
        fp.write("%s\n" % gene)
    fp.close()

    if not os.path.exists(os.path.join(results_dir, Path('selected_features'))):
        os.mkdir(os.path.join(results_dir, Path('selected_features')))

    extracted_features_training = os.path.join(results_dir, Path('selected_features') / 'training')
    if not os.path.exists(extracted_features_training):
        os.mkdir(extracted_features_training)

    extracted_features_test = os.path.join(results_dir, Path('selected_features') / 'test')
    if not os.path.exists(extracted_features_test):
        os.mkdir(extracted_features_test)

    for index, row in X_train[selected_genes].iterrows():
        row = np.asarray(row)
        np.save(os.path.join(extracted_features_training, index + '.npy'), row)

    for index, row in X_test[selected_genes].iterrows():
        row = np.asarray(row)
        np.save(os.path.join(extracted_features_test, index + '.npy'), row)

    print(">> training features saved to " + extracted_features_training)
    print(">> testing features saved to " + extracted_features_test)


def read_config_file(config_file_path, section):
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file_path)

    if section == 'welch_t':
        params['alpha'] = config.getfloat('welch_t_test', 'alpha')

    elif section == 'svm_t_rfe':
        params['alpha'] = config.getfloat('svm_t_rfe', 'alpha')
        params['theta'] = config.getfloat('svm_t_rfe', 'theta')
        params['cv_grid_search_rank'] = config.getint('svm_t_rfe', 'cv_grid_search_rank')
        params['cv_grid_search_acc'] = config.getint('svm_t_rfe', 'cv_grid_search_acc')
        params['cv_outer'] = config.getint('svm_t_rfe', 'cv_outer')
        params['top_ranked'] = config.getint('svm_t_rfe', 'top_ranked')
        params['t_stat_threshold'] = config.getfloat('svm_t_rfe', 't_stat_threshold')
        if config['svm_t_rfe']['kernel'] == 'linear' or config['svm_t_rfe']['kernel'] == 'rbf':
            params['kernel'] = config['svm_t_rfe']['kernel']
        else:
            sys.stderr.write("Invalid value for <kernel> in config file")
            exit(1)

    elif section == 'svm':
        params['cv_grid_search_acc'] = config.getint('svm', 'cv_grid_search_acc')
    else:
        sys.stderr.write("Invalid value for <section> in config file")
        exit(1)

    return params


def eval_asymmetry_and_kurt(df):
    n_skew_pos = 0
    n_skew_neg = 0
    n_kurt_1 = 0
    n_kurt_2 = 0

    for gene in tqdm(df.columns, desc="Evaluate asymmetry and kurt...", file=sys.stdout):
        if stats.skew(df[gene]) > 0.5:
            n_skew_pos += 1
        elif stats.skew(df[gene]) < -0.5:
            n_skew_neg += 1
        if stats.kurtosis(df[gene]) > 0:
            n_kurt_1 += 1
        elif stats.kurtosis(df[gene]) < 0:
            n_kurt_2 += 1

    print(">> Percentage of genes with asymmetric distribution (verso sx): %.3f" % (100 * (n_skew_pos / len(df.columns))))
    print(">> Percentage of genes with asymmetric distribution (verso dx): %.3f" % (100 * (n_skew_neg / len(df.columns))))
    print(">> Percentage of genes with platykurtic distribution: %.3f" % (100 * (n_kurt_2 / len(df.columns))))
    print(">> Percentage of genes with leptokurtic distribution: %.3f" % (100 * (n_kurt_1 / len(df.columns))))
    return n_skew_pos, n_skew_neg, n_kurt_1, n_kurt_2
