import configparser
import os
import statistics
import sys
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
    return data_frame_0, data_frame_1


def eval_asymmetry_and_kurt(df):
    n_skew_pos = 0
    n_skew_neg = 0
    n_kurt_1 = 0
    n_kurt_2 = 0

    for gene in tqdm(df.columns, desc=">> Evaluate asymmetry and kurt...", file=sys.stdout):
        if stats.skew(df[gene]) > 0.5:
            n_skew_pos += 1
        elif stats.skew(df[gene]) < -0.5:
            n_skew_neg += 1
        if stats.kurtosis(df[gene]) > 0:
            n_kurt_1 += 1
        elif stats.kurtosis(df[gene]) < 0:
            n_kurt_2 += 1

    return n_skew_pos, n_skew_neg, n_kurt_1, n_kurt_2


def read_config_file(config_file_path, features_extraction_method):
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file_path)

    if features_extraction_method == 'pca':
        params['percentage_of_variance'] = config.getfloat('pca', 'percentage_of_variance')

    elif features_extraction_method == 'welch_t_pca':
        params['alpha'] = config.getfloat('welch_t_test', 'alpha')
        params['percentage_of_variance'] = config.getfloat('pca', 'percentage_of_variance')

    elif features_extraction_method == 'welch_t':
        params['alpha'] = config.getfloat('welch_t_test', 'alpha')

    elif features_extraction_method == 'svm_t_rfe':
        params['alpha'] = config.getfloat('welch_t_test', 'alpha')
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
    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)

    return params


def tsne_pca(X_train, y_train):
    pca_50 = PCA(n_components=50, random_state=42)
    pca_result_50 = pca_50.fit_transform(X_train)
    print('Cumulative explained variation for 50 principal components: {}'.format(
        np.sum(pca_50.explained_variance_ratio_)))

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=42)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    df_tsne_pca = pd.DataFrame(
        dict(tsne_pca50_one=tsne_pca_results[:, 0], tsne_pca50_two=tsne_pca_results[:, 1], y=y_train))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="tsne_pca50_one", y="tsne_pca50_two",
        hue="y_train",
        palette=sns.color_palette("hls", 2),
        data=df_tsne_pca,
        legend="full",
        alpha=0.3,
    )
    plt.xlabel('tsne_pca50_one')
    plt.ylabel('tsne_pca50_two')
    plt.show()