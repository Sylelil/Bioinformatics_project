import argparse
import math
import os
import sys
from os import path
from pathlib import Path

from config import paths
from genes import methods
from genes.features_selection_methods.svm_t_rfe import genes_selection_svm_t_rfe
from genes.features_selection_methods.welch_t import genes_selection_welch_t

from src.data_manipulation import split_data


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

    if not os.path.exists(paths.genes_dir):
        sys.stderr.write(f'{paths.genes_dir} does not exists')
        exit(2)

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    if not os.path.exists(paths.split_data_dir):
        sys.stderr.write(f'{paths.split_data_dir} does not exists')
        exit(2)

    if not os.path.exists(paths.welch_t_results_dir):
        os.makedirs(paths.welch_t_results_dir)

    if not os.path.exists(paths.svm_t_rfe_results_dir):
        os.makedirs(paths.svm_t_rfe_results_dir)

    if not os.path.exists(paths.genes_config_dir):
        os.makedirs(paths.genes_config_dir)

    if not os.path.exists(paths.welch_t_selected_features_train):
        os.makedirs(paths.welch_t_selected_features_train)

    if not os.path.exists(paths.welch_t_selected_features_test):
        os.makedirs(paths.welch_t_selected_features_test)

    if not os.path.exists(paths.svm_t_rfe_selected_features_train):
        print("ok")
        os.makedirs(paths.svm_t_rfe_selected_features_train)

    if not os.path.exists(paths.svm_t_rfe_selected_features_test):
        os.makedirs(paths.svm_t_rfe_selected_features_test)

    if not os.path.exists(paths.svm_t_rfe_selected_features_val):
        os.makedirs(paths.svm_t_rfe_selected_features_val)

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.method)

    print("\nReading split gene expression data:")
    genes_splits_path = Path(paths.split_data_dir) / 'genes'
    if not os.path.exists(paths.split_data_dir):
        print("%s not existing." % paths.split_data_dir)
        exit()
    if not os.path.exists(genes_splits_path):
        print("%s not existing." % genes_splits_path)
        exit()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data.get_genes_split_data(genes_splits_path, val_data=True)

    X_train_val = X_train.append(X_val, sort=False)
    y_train_val = y_train + y_val

    print("\nExploratory analysis:")
    # Compute number of samples
    X_train_val_0 = X_train_val.loc[X_train_val.index.str.endswith('_0')]
    X_train_val_1 = X_train_val.loc[X_train_val.index.str.endswith('_1')]
    print(f'>> Train + val (Tot training) data:\n>> Tot = {len(X_train_val)}\n'
          f'>> Tumor samples = {len(X_train_val_1)}\n>> Normal samples = {len(X_train_val_0)}\n')

    X_test_0 = X_test.loc[X_test.index.str.endswith('_0')]
    X_test_1 = X_test.loc[X_test.index.str.endswith('_1')]
    print(f'>> Test data:\n>> Tot = {len(X_test)}\n'
          f'>> Tumor samples = {len(X_test_1)}\n>> Normal samples = {len(X_test_0)}\n')

    # Compute number of features
    n_features = len(X_train_val.columns)
    print(f">> Number of features (genes): {n_features}\n")

    # Evaluate normality by skewness and kourt
    methods.eval_asymmetry_and_kurt(X_train_val)

    # Apply logarithmic transformation on gene expression data
    # Description : x = Log(x+1), where x is the gene expression value
    print(f'\nLogarithmic transformation on gene expression data:'
          f'\n>> Computing logarithmic transformation...')
    X_train = X_train.applymap(lambda x: math.log(x + 1, 10))
    X_val = X_val.applymap(lambda x: math.log(x + 1, 10))
    X_test = X_test.applymap(lambda x: math.log(x + 1, 10))
    X_train_val = X_train.append(X_val, sort=False)
    print(">> Done")

    print("\nDifferentially gene expression analysis [DGEA]")
    if args.method == 'welch_t':
        # feature selection
        selected_genes = genes_selection_welch_t(X_train_val, params, paths.welch_t_results_dir)

        # save selected genes on file
        selected_genes_file = str(params['alpha']) + "_selected_genes.txt"
        selected_genes_path = os.path.join(paths.welch_t_results_dir, selected_genes_file)
        fp = open(selected_genes_path, "w")
        for gene in selected_genes:
            fp.write("%s\n" % gene)
        fp.close()

        # saving selected features
        print("\nSaving selected training gene features on disk...")
        methods.save_selected_genes(X_train[selected_genes], paths.welch_t_selected_features_train)

        print("\nSaving selected validation gene features on disk...")
        methods.save_selected_genes(X_val[selected_genes], paths.welch_t_selected_features_val)

        print("\nSaving selected test gene features on disk...")
        methods.save_selected_genes(X_test[selected_genes], paths.welch_t_selected_features_test)

    elif args.method == 'svm_t_rfe':
        # feature selection
        selected_genes = genes_selection_svm_t_rfe(X_train_val, y_train_val, params, paths.svm_t_rfe_results_dir, paths.genes_config_dir)

        # save selected genes on file
        selected_genes_file = str(params['alpha']) + "_" + str(params['t_stat_threshold']) + "_" + \
                              str(params['theta']) + "_" + str(params['cv_grid_search_rank']) + "_" + \
                              params['scoring_name'] + "_selected_genes.txt"

        selected_genes_path = os.path.join(paths.svm_t_rfe_results_dir, selected_genes_file)
        fp = open(selected_genes_path, "w")
        for gene in selected_genes:
            fp.write("%s\n" % gene)
        fp.close()

        # saving selected features
        print("\nSaving selected training gene features on disk...")
        methods.save_selected_genes(X_train[selected_genes], paths.svm_t_rfe_selected_features_train)

        print("\nSaving selected validation gene features on disk...")
        methods.save_selected_genes(X_val[selected_genes], paths.svm_t_rfe_selected_features_val)

        print("\nSaving selected test gene features on disk...")
        methods.save_selected_genes(X_test[selected_genes], paths.svm_t_rfe_selected_features_test)

    else:
        sys.stderr.write("Invalid value for <feature extraction method>")
        exit(1)


if __name__ == "__main__":
    main()
