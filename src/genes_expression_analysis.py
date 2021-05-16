import argparse
import math
import os
import sys
from pathlib import Path
from numpy import mean, std, median

from config import paths
from config.paths import BASE_DIR
from genes import utils
from genes.features_selection_method.svm_t_rfe_no_pipe import genes_selection_svm_t_rfe
from src.common import split_data, plots
import matplotlib.pyplot as plt


def main():
    """
         Description: Main performing preprocessing steps in order to select
            most relevant features from gene expression data
    """
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)

    parser.add_argument('--method',
                        help='Feature selection method',
                        choices=['svm_t_rfe'],
                        required=True,
                        default='svm_t_rfe',
                        type=str)

    args = parser.parse_args()

    if not os.path.exists(paths.genes_dir):
        sys.stderr.write(f'{paths.genes_dir} does not exists')
        exit(2)

    if not os.path.exists(args.cfg) or (not os.path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    if not os.path.exists(paths.split_data_dir):
        sys.stderr.write(f'{paths.split_data_dir} does not exists')
        exit(2)

    # config folder
    genes_config_dir = BASE_DIR / 'config' / 'genes'  # Directory di configurazione per i geni

    if not os.path.exists(paths.svm_t_rfe_results_dir):
        os.makedirs(paths.svm_t_rfe_results_dir)

    if not os.path.exists(genes_config_dir):
        os.makedirs(genes_config_dir)

    if not os.path.exists(paths.svm_t_rfe_selected_features_train):
        os.makedirs(paths.svm_t_rfe_selected_features_train)

    if not os.path.exists(paths.svm_t_rfe_selected_features_test):
        os.makedirs(paths.svm_t_rfe_selected_features_test)

    if not os.path.exists(paths.svm_t_rfe_selected_features_val):
        os.makedirs(paths.svm_t_rfe_selected_features_val)

    # Read configuration file
    params = utils.read_config_file(args.cfg, args.method)

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
    X_train_0 = X_train.loc[X_train.index.str.endswith('_0')]
    X_train_1 = X_train.loc[X_train.index.str.endswith('_1')]
    print(f'>> Train data:\n>> Tot = {len(X_train)}\n'
          f'>> Tumor samples = {len(X_train_1)}\n>> Normal samples = {len(X_train_0)}\n')

    X_val_0 = X_val.loc[X_val.index.str.endswith('_0')]
    X_val_1 = X_val.loc[X_val.index.str.endswith('_1')]
    print(f'>> Val data:\n>> Tot = {len(X_val)}\n'
          f'>> Tumor samples = {len(X_val_1)}\n>> Normal samples = {len(X_val_0)}\n')

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
    print(f">> Number of features (genes): {n_features}")

    # Apply logarithmic transformation on gene expression data
    # Description : x = Log(x+1), where x is the gene expression value
    print(f'\nLogarithmic transformation on gene expression data:'
          f'\n>> Computing logarithmic transformation...')
    X_train_log = X_train.applymap(lambda x: math.log(x + 1, 10))
    X_val_log = X_val.applymap(lambda x: math.log(x + 1, 10))
    X_test_log = X_test.applymap(lambda x: math.log(x + 1, 10))
    X_train_val_log = X_train_log.append(X_val_log, sort=False)
    print(">> Done")

    print("\nDifferentially gene expression analysis [DGEA]")

    if args.method == 'svm_t_rfe':
        # feature selection
        selected_genes = genes_selection_svm_t_rfe(X_train_val_log, y_train_val, params, paths.svm_t_rfe_results_dir,
                                                   genes_config_dir)
        # save selected genes on file
        fp = open(paths.svm_t_rfe_results_dir / "selected_genes.txt", "w")
        for gene in selected_genes:
            fp.write("%s\n" % gene)
        fp.close()

        plots.plot_features_box_plots(X_train_val[selected_genes[:4]], y_train_val, paths.svm_t_rfe_results_dir)

        # saving selected features
        print("\nSaving selected training gene features on disk...")
        utils.save_selected_genes(X_train_log[selected_genes], paths.svm_t_rfe_selected_features_train)

        print("\nSaving selected validation gene features on disk...")
        utils.save_selected_genes(X_val_log[selected_genes], paths.svm_t_rfe_selected_features_val)

        print("\nSaving selected test gene features on disk...")
        utils.save_selected_genes(X_test_log[selected_genes], paths.svm_t_rfe_selected_features_test)

    else:
        sys.stderr.write("Invalid value for <feature selection method>")
        exit(1)


if __name__ == "__main__":
    main()
