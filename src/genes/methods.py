import configparser
import os
import sys
from pathlib import Path

import imblearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer
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


def save_selected_genes(X, selected_genes, extracted_features_dir):

    for index, row in X[selected_genes].iterrows():
        row = np.asarray(row)
        np.save(os.path.join(extracted_features_dir, index + '.npy'), row)

    print(">> Features saved to " + extracted_features_dir)


def read_config_file(config_file_path, section):
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file_path)

    if section == 'welch_t':
        params['alpha'] = config.getfloat('welch_t_test', 'alpha')

    elif section == 'svm_t_rfe':
        params['random_state'] = config.getint('svm_t_rfe', 'random_state')
        params['alpha'] = config.getfloat('svm_t_rfe', 'alpha')
        params['theta'] = config.getfloat('svm_t_rfe', 'theta')
        params['cv_grid_search_rank'] = config.getint('svm_t_rfe', 'cv_grid_search_rank')
        params['cv_grid_search_acc'] = config.getint('svm_t_rfe', 'cv_grid_search_acc')
        params['cv_outer'] = config.getint('svm_t_rfe', 'cv_outer')
        params['top_ranked'] = config.getint('svm_t_rfe', 'top_ranked')
        params['t_stat_threshold'] = config.getfloat('svm_t_rfe', 't_stat_threshold')
        if config['svm_t_rfe']['scoring'] == 'accuracy':
            params['scoring'] = make_scorer(metrics.accuracy_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'matt_coef':
            params['scoring'] = make_scorer(metrics.matthews_corrcoef)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'recall':
            params['scoring'] = make_scorer(metrics.recall_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'precision':
            params['scoring'] = make_scorer(metrics.precision_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'f1_score':
            params['scoring'] = make_scorer(metrics.f1_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'sensitivity':
            params['scoring'] = make_scorer(imblearn.metrics.sensitivity_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'specificity':
            params['scoring'] = make_scorer(imblearn.metrics.specificity_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        else:
            sys.stderr.write("Invalid value for <scoring> in config file")
            exit(1)
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


def pca(X_train, y_train):

    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X_train)
    print('Cumulative explained variation for 2 principal components: {}'.format(
        np.sum(pca.explained_variance_ratio_)))

    df_pca = pd.DataFrame(
        dict(pca_one=pca_results[:, 0], pca_two=pca_results[:, 1], y=y_train))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="pca_one", y="pca_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_pca,
        legend="full",
        alpha=0.3,
    )
    plt.xlabel('pca_one')
    plt.ylabel('pca_two')
    plt.show()


def pca_3D(X_train, y_train):

    pca = PCA(n_components=3, random_state=42)
    pca_results = pca.fit_transform(X_train)
    print('Cumulative explained variation for 3 principal components: {}'.format(
        np.sum(pca.explained_variance_ratio_)))

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    scatter = ax.scatter(
        xs=pca_results[:, 0],
        ys=pca_results[:, 1],
        zs=pca_results[:, 2],
        c=y_train,
        cmap='tab10'
    )
    legend = ax.legend(*scatter.legend_elements())
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    ax.add_artist(legend)

    plt.show()


def tsne_pca(X_train, y_train):

    pca_50 = PCA(n_components=50, random_state=42)
    pca_result_50 = pca_50.fit_transform(X_train)
    print('Cumulative explained variation for 50 principal components: {}'.format(
        np.sum(pca_50.explained_variance_ratio_)))

    tsne = TSNE(n_components=2, random_state=42)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    df_tsne_pca = pd.DataFrame(
        dict(tsne_pca50_one=tsne_pca_results[:, 0], tsne_pca50_two=tsne_pca_results[:, 1], y=y_train))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="tsne_pca50_one", y="tsne_pca50_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_tsne_pca,
        legend="full",
        alpha=0.3,
    )
    plt.xlabel('tsne_pca50_one')
    plt.ylabel('tsne_pca50_two')
    plt.show()


def tsne(X_train, y_train):
    # , verbose=0, perplexity=40, n_iter=300,

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X_train)

    df_tsne = pd.DataFrame(
        dict(tsne_one=tsne_results[:, 0], tsne_two=tsne_results[:, 1], y=y_train))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="tsne_one", y="tsne_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_tsne,
        legend="full",
        alpha=0.3,
    )
    plt.xlabel('tsne_one')
    plt.ylabel('tsne_two')
    plt.show()


def tsne_3D(X_train, y_train):

    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(X_train)

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    scatter = ax.scatter(
        xs=tsne_results[:, 0],
        ys=tsne_results[:, 1],
        zs=tsne_results[:, 2],
        c=y_train,
        cmap='tab10'
    )
    legend = ax.legend(*scatter.legend_elements())
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    ax.add_artist(legend)

    plt.show()
