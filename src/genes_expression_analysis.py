import argparse
import math
import os
import sys
from collections import Counter
from os import path
from pathlib import Path
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.preprocessing import StandardScaler
from genes import methods
from genes.features_selection_methods.svm_t_rfe import genes_selection_svm_t_rfe
from genes.features_selection_methods.welch_t import genes_selection_welch_t
import pandas as pd

from common import split_data


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)

    parser.add_argument('--method',
                        help='Feature extraction method',
                        choices=['welch_t', 'svm_t_rfe'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    welch_t_results_dir = Path('results') / 'genes' / 'welch_t'
    svm_t_rfe_results_dir = Path('results') / 'genes' / 'svm_t_rfe'
    splits_dir = Path('assets') / 'data_splits'
    config_dir = Path('config') / 'genes'
    path_genes = Path('datasets') / 'genes'

    if not os.path.exists(path_genes):
        sys.stderr.write(f'{path_genes} does not exists')
        exit(2)

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    if not os.path.exists(splits_dir):
        sys.stderr.write(f'{splits_dir} does not exists')
        exit(2)

    if not os.path.exists(Path('config')):
        os.mkdir(Path('config'))

    if not os.path.exists(Path('results')):
        os.mkdir(Path('results'))

    if not os.path.exists(Path('results') / 'genes'):
        os.mkdir(Path('results'))

    if not os.path.exists(welch_t_results_dir):
        os.mkdir(welch_t_results_dir)

    if not os.path.exists(svm_t_rfe_results_dir):
        os.mkdir(svm_t_rfe_results_dir)

    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.method)

    print("\nReading gene expression data:")
    X_train, X_test, y_train, y_test = split_data.get_genes_split_data(path_genes, splits_dir)

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
    n_features = len(X_train.columns)
    print(f">> Number of features (genes): {n_features}\n")

    # Evaluate normality by skewness and kourt
    methods.eval_asymmetry_and_kurt(X_train)

    # Apply logarithmic transformation on gene expression data
    # Description : x = Log(x+1), where x is the gene expression value
    print(f'\nLogarithmic transformation on gene expression data:'
          f'\n>> Computing logarithmic transformation...')
    X_train = X_train.applymap(lambda x: math.log(x + 1, 10))
    X_test = X_test.applymap(lambda x: math.log(x + 1, 10))
    print(">> Done")

    print("\nStandard scaler on gene expression data:")
    scaler = StandardScaler()
    train_scaled_features = scaler.fit_transform(X_train.values)
    X_train = pd.DataFrame(train_scaled_features, index=X_train.index, columns=X_train.columns)
    print(">> Training features scaled")

    test_scaled_features = scaler.transform(X_test.values)
    X_test = pd.DataFrame(test_scaled_features, index=X_test.index, columns=X_test.columns)
    print(">> Test features scaled")

    # SMOTE
    print("\nSMOTE")
    print(">> Oversampling training data")
    sm = SMOTE(sampling_strategy=1.0, random_state=42, n_jobs=-1)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(Counter(y_train_sm))
    X_train_sm["target"] = np.array([str(x) for x in y_train_sm])
    X_train_sm.index = list(X_train_sm["target"])
    del X_train_sm["target"]
    print(X_train_sm)

    print("\nDifferentially gene expression analysis [DGEA]")
    if args.method == 'welch_t':
        selected_genes = genes_selection_welch_t(X_train, params, welch_t_results_dir)
        selected_genes_file = str(params['alpha']) + "_" + str(params['t_stat_threshold']) + "_" + \
                              str(params['theta']) + "_" + str(params['cv_grid_search_rank']) + \
                              "_welch_t_selected_genes.txt"
        selected_genes_path = os.path.join(welch_t_results_dir, selected_genes_file)
        print("\nSaving selected gene features on disk...")
        methods.save_selected_genes(X_train[selected_genes], X_test[selected_genes], selected_genes,
                                    welch_t_results_dir, selected_genes_path)
    elif args.method == 'svm_t_rfe':

        selected_genes = genes_selection_svm_t_rfe(X_train_sm, y_train_sm, params, svm_t_rfe_results_dir, config_dir)
        selected_genes_file = str(params['alpha']) + "_" + str(params['t_stat_threshold']) + "_" + \
                              str(params['theta']) + "_" + str(params['cv_grid_search_rank']) + \
                              "_svm_t_rfe_selected_genes.txt"
        selected_genes_path = os.path.join(svm_t_rfe_results_dir, selected_genes_file)
        print("\nSaving selected gene features on disk...")
        methods.save_selected_genes(X_train[selected_genes], X_test[selected_genes], selected_genes,
                                    svm_t_rfe_results_dir, selected_genes_path)
    else:
        sys.stderr.write("Invalid value for <feature extraction method>")
        exit(1)


if __name__ == "__main__":
    main()
