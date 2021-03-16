import argparse
import os
import sys
from os import path
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, \
    plot_confusion_matrix, plot_roc_curve, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.genes import methods
from src.genes.features_selection_methods.pca import genes_extraction_pca
from src.genes.features_selection_methods.svm_t_rfe import genes_selection_svm_t_rfe
from src.genes.features_selection_methods.welch_t import genes_selection_welch_t
from src.genes.features_selection_methods.welch_t_pca import genes_extraction_welch_t_pca
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)

    parser.add_argument('--method',
                        help='Feature extraction method',
                        choices=['pca', 'svm_t_rfe', 'welch_t', 'welch_t_pca'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.method)

    path_genes = Path('datasets') / 'genes'
    if not os.path.exists(path_genes):
        sys.stderr.write(f'{path_genes} does not exists')
        exit(2)

    print("Reading gene expression data:")
    df_normal, df_tumor = methods.read_gene_expression_data(path_genes)  # normal = 0, tumor = 1
    df_patients = df_normal.append(df_tumor, sort=False)  # Merge normal data frame with tumor data frame

    # divide dataset in training and test
    y = np.array([int(x[-1:]) for x in df_patients.index])
    print(type(y[0]))
    X_train, X_test, y_train, y_test = train_test_split(df_patients, y, train_size=0.70, random_state=42, shuffle=True)

    print("\nExploratory analysis:")
    # Compute number of samples
    X_train_0 = X_train.loc[X_train.index.str.endswith('_0')]
    X_train_1 = X_train.loc[X_train.index.str.endswith('_1')]
    print(f'>> Training data:\n>> Tot = {len(X_train)}\n'
          f'>> Tumor samples = {len(X_train_1)}\n>> Normal samples = {len(X_train_0)}\n')

    X_test_0 = X_test.loc[X_test.index.str.endswith('_0')]
    X_test_1 = X_test.loc[X_test.index.str.endswith('_1')]
    print(f'>> Test data:\n>> Tot = {len(X_test)}\n'
          f'>> Tumor samples = {len(X_test_1)}\n>> Normal samples = {len(X_test_0)}\n')

    # Compute number of features
    n_features = len(df_patients.columns)
    print(f">> Number of features (genes): {n_features}")
    print(df_patients)

    # Evaluate normality by skewness and kourt
    '''
    n_skew_pos, n_skew_neg, n_kurt_1, n_kurt_2 = methods.eval_asymmetry_and_kurt(X_train)

    print("Percentage of genes with asymmetric distribution (verso sx): %.3f" % (100 * (n_skew_pos / n_features)))
    print("Percentage of genes with asymmetric distribution (verso dx): %.3f" % (100 * (n_skew_neg / n_features)))
    print("Percentage of genes with platykurtic distribution: %.3f" % (100 * (n_kurt_2 / n_features)))
    print("Percentage of genes with leptokurtic distribution: %.3f" % (100 * (n_kurt_1 / n_features)))
    '''
    # Grafico a torta di skew and kurtosys
    # TODO

    #methods.tsne_pca(X_train, y_train)
    '''
    print(X_train.shape)
    print(X_test.shape)
    print("\nCompute accuracy on test set considering all features [SVM classifier]")
    pipe_grid = Pipeline([('svm', SVC(kernel=params['kernel']))])
    cv = KFold(n_splits=params['cv_grid_search_rank'])
    grid = GridSearchCV(estimator=pipe_grid, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=cv,
                        refit=True)
    svm.fit(X_train.to_numpy(), y_train)
    pred = svm.predict(X_test.to_numpy())
    y_score = svm.decision_function(X_test.to_numpy())
    print("accuracy= %f" % accuracy_score(y_test, pred))
    average_precision = average_precision_score(y_test, y_score, pos_label=0)
    precision = precision_score(y_test, pred, pos_label=0)
    recall = recall_score(y_test, pred, pos_label=0)
    print('Average precision-recall score: %f' % average_precision)
    print('Precision score: %f' % precision)
    print('Recall score: %f' % recall)
    plot_confusion_matrix(svm, X_test.to_numpy(), y_test)
    plt.show()
    plot_roc_curve(svm, X_test.to_numpy(), y_test, pos_label=0)
    plt.show()
    '''

    print("\nDifferentially gene expression analysis [DGEA]")
    if args.method == 'pca':
        genes_extraction_pca(X_train, params)
    elif args.method == 'welch_t_pca':
        genes_extraction_welch_t_pca(X_train, params)
    elif args.method == 'welch_t':
        genes_selection_welch_t(X_train, params)
    elif args.method == 'svm_t_rfe':
        results_dir = Path('results') / 'genes'
        config_dir = Path('config') / 'genes'
        if not os.path.exists(config_dir):
            os.mkdir(config_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        selected_features, C = genes_selection_svm_t_rfe(X_train, y_train, params, results_dir, config_dir)
        print(selected_features)
        X_train_reduced = X_train[selected_features].to_numpy()
        X_test_reduced = X_test[selected_features].to_numpy()
        svm = SVC(kernel=params['kernel'], C=C)
        svm.fit(X_train_reduced, y_train)
        pred = svm.predict(X_test_reduced)
        print("accuracy= %f" % accuracy_score(y_test, pred))
        average_precision = average_precision_score(y_test, pred, pos_label=0)
        precision = precision_score(y_test, pred, pos_label=0)
        recall = recall_score(y_test, pred, pos_label=0)
        print('Average precision-recall score: %f' % average_precision)
        print('Precision score: %f' % precision)
        print('Recall score: %f' % recall)
        plot_confusion_matrix(svm, X_test_reduced, y_test, pos_label=0)
        plt.show()
        plot_roc_curve(svm, X_test_reduced, y_test, pos_label=0)
        plt.show()

    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == "__main__":
    main()
