import argparse
import math
import os
import sys
from collections import Counter
from os import path
from pathlib import Path
import seaborn as sns
from imblearn.metrics import classification_report_imbalanced, sensitivity_score, specificity_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, \
    plot_confusion_matrix, plot_roc_curve, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from genes import methods
from genes.features_selection_methods.pca import genes_extraction_pca
from genes.features_selection_methods.svm_t_rfe import genes_selection_svm_t_rfe
from genes.features_selection_methods.welch_t import genes_selection_welch_t
from genes.features_selection_methods.welch_t_pca import genes_extraction_welch_t_pca
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

    results_dir = Path('results') / 'genes'
    extracted_features_training = Path('results') / 'genes' / 'extracted_features' / 'training'
    extracted_features_test = Path('results') / 'genes' / 'extracted_features' / 'test'
    config_dir = Path('config') / 'genes'
    path_genes = Path('datasets') / 'genes'
    path_to_csv_normal = Path('datasets') / 'csv' / 'normal'
    path_to_csv_tumor = Path('datasets') / 'csv' / 'tumor'

    if not os.path.exists(path_genes):
        sys.stderr.write(f'{path_genes} does not exists')
        exit(2)

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if not os.path.exists(Path('datasets') / 'csv'):
        os.mkdir(Path('datasets') / 'csv')

    if not os.path.exists(Path('results') / 'genes' / 'extracted_features'):
        os.mkdir(Path('results') / 'genes' / 'extracted_features')

    if not os.path.exists(path_to_csv_normal):
        os.mkdir(path_to_csv_normal)

    if not os.path.exists(path_to_csv_tumor):
        os.mkdir(path_to_csv_tumor)

    if not os.path.exists(extracted_features_training):
        os.mkdir(extracted_features_training)

    if not os.path.exists(extracted_features_test):
        os.mkdir(extracted_features_test)

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.method)

    print("Reading gene expression data:")
    df_normal, df_tumor = methods.read_gene_expression_data(path_genes)  # normal = 0, tumor = 1
    df_patients = df_normal.append(df_tumor, sort=False)  # Merge normal data frame with tumor data frame

    #df_normal.to_csv(os.path.join(path_to_csv_normal, "normal.csv"))
    #df_tumor.to_csv(os.path.join(path_to_csv_tumor, "tumor.csv"))

    #with open(os.path.join(path_to_csv_normal, "normal.txt"), 'w') as outfile:
    #    df_normal.to_string(outfile)

    # divide dataset in training and test
    y = np.array([int(x[-1:]) for x in df_patients.index])
    X_train, X_test, y_train, y_test = train_test_split(df_patients, y, test_size=0.30, random_state=42, shuffle=True)

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
    '''
    # Evaluate normality by skewness and kourt
    n_skew_pos, n_skew_neg, n_kurt_1, n_kurt_2 = methods.eval_asymmetry_and_kurt(X_train)

    print("Percentage of genes with asymmetric distribution (verso sx): %.3f" % (100 * (n_skew_pos / n_features)))
    print("Percentage of genes with asymmetric distribution (verso dx): %.3f" % (100 * (n_skew_neg / n_features)))
    print("Percentage of genes with platykurtic distribution: %.3f" % (100 * (n_kurt_2 / n_features)))
    print("Percentage of genes with leptokurtic distribution: %.3f" % (100 * (n_kurt_1 / n_features)))
    '''

    # Grafico a torta di skew and kurtosys
    # TODO

    print(X_train)
    # TODO: SMOTE
    print("\n[SMOTE]")
    sm = SMOTE(sampling_strategy=1.0, random_state=42, n_jobs=-1)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(Counter(y_train_sm))
    X_train_sm["target"] = np.array([str(x) for x in y_train_sm])
    print(list(X_train_sm["target"]))
    X_train_sm.index = list(X_train_sm["target"])
    del X_train_sm["target"]
    print(X_train_sm)

    print("\nDifferentially gene expression analysis [DGEA]")
    if args.method == 'pca':
        genes_extraction_pca(X_train, params)
    elif args.method == 'welch_t_pca':
        genes_extraction_welch_t_pca(X_train, params)
    elif args.method == 'welch_t':
        genes_selection_welch_t(X_train, params)
    elif args.method == 'svm_t_rfe':

        selected_genes = genes_selection_svm_t_rfe(X_train_sm, y_train_sm, params, results_dir, config_dir)

        selected_genes_path = os.path.join(results_dir, "selected_genes.txt")
        fp = open(selected_genes_path, "w")
        for gene in selected_genes:
            fp.write("%s\n" % gene)
        fp.close()

        for index, row in X_train[selected_genes].iterrows():
            row = np.asarray(row)
            print(row.shape)
            np.save(os.path.join(extracted_features_training, index + '.npy'), row)

        for index, row in X_test[selected_genes].iterrows():
            row = np.asarray(row)
            print(row.shape)
            np.save(os.path.join(extracted_features_test, index + '.npy'), row)

        # 2.a.2 Apply logarithmic transformation on gene expression data
        #       Description : x = Log(x+1), where x is the gene expression value
        print(f'\n[DGEA pre-processing] Logarithmic transformation on gene expression data:'
              f'\n>> Computing logarithmic transformation...')
        X_train_sm = X_train_sm.applymap(lambda x: math.log(x + 1, 10))
        X_test = X_test.applymap(lambda x: math.log(x + 1, 10))

        tuned_parameters = dict(svm__C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
        X_train_reduced = X_train_sm[selected_genes].to_numpy()
        X_test_reduced = X_test[selected_genes].to_numpy()

        pipe_grid = Pipeline([('svm', SVC(kernel='linear'))])
        cv = KFold(n_splits=params['cv_grid_search_acc'])
        clf = GridSearchCV(estimator=pipe_grid, param_grid=tuned_parameters, scoring='accuracy', cv=cv, n_jobs=-1, refit=True)
        clf.fit(X_train_reduced, y_train_sm)
        pred = clf.predict(X_test_reduced)
        print("accuracy= %f" % accuracy_score(y_test, pred))
        average_precision = average_precision_score(y_test, pred, pos_label=0)
        precision = precision_score(y_test, pred, average='binary', pos_label=0)
        recall = recall_score(y_test, pred, average='binary', pos_label=0)
        sensitivity = sensitivity_score(y_test, pred, average='binary', pos_label=0)
        specificity = specificity_score(y_test, pred, average='binary', pos_label=0)
        print("\npos_label = 0")
        print('Average precision-recall score: %f' % average_precision)
        print('Precision score: %f' % precision)
        print('Recall score: %f' % recall)
        print('sensitivity: %f' % sensitivity)
        print('specificity: %f' % specificity)

        average_precision_1 = average_precision_score(y_test, pred, pos_label=1)
        precision_1 = precision_score(y_test, pred, average='binary', pos_label=1)
        recall_1 = recall_score(y_test, pred, average='binary', pos_label=1)
        sensitivity_1 = sensitivity_score(y_test, pred, average='binary', pos_label=1)
        specificity_1 = specificity_score(y_test, pred, average='binary', pos_label=1)
        print("\npos_label = 1")
        print('Average precision-recall score: %f' % average_precision_1)
        print('Precision score: %f' % precision_1)
        print('Recall score: %f' % recall_1)
        print('sensitivity: %f' % sensitivity_1)
        print('specificity: %f' % specificity_1)

        print(classification_report_imbalanced(y_test, pred))
        plot_confusion_matrix(clf, X_test_reduced, y_test)
        plt.show()
        plot_roc_curve(clf, X_test_reduced, y_test)
        plt.show()

    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == "__main__":
    main()
