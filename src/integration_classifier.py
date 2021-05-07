import argparse
import os
from pathlib import Path
from config import paths
from src.integration import utils
from src.integration.classification_methods import nn_classification, shallow_classification
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def args_parse():
    """
       Description: Parse command-line arguments.
       :returns: arguments parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file path',
                        required=True,
                        type=str)
    parser.add_argument('--data',
                        help='Data path',
                        required=True,
                        type=str)
    parser.add_argument('--classification_method',
                        help='Classification method',
                        choices=['linearsvc', 'sgd', 'nn', 'pca_nn'],
                        required=False,
                        type=str)
    parser.add_argument('--balancing',
                        help='Class balancing method',
                        choices=['random_upsampling', 'combined', 'smote', 'weights'],
                        required=False,
                        type=str)
    parser.add_argument('--n_principal_components',
                        help='Number of Principal Components for PCA',
                        required=False,
                        type=int)
    parser.add_argument('--plot_explained_variance',
                        help='Either to plot explained variance to choose best number of Principal Components or not',
                        required=False,
                        action='store_true')

    args = parser.parse_args()
    return args


def main():
    """
       Description: Train and test classifier on concatenated features, with possible preprocessing techniques and class balancing.
    """
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    params = utils.read_config_file(args.cfg, args.classification_method)

    data_path = args.data
    concatenated_results_path = paths.concatenated_results_dir
    train_filepath = Path(concatenated_results_path) / 'train' / 'concat_data.csv'
    val_filepath = Path(concatenated_results_path) / 'val' / 'concat_data.csv'
    test_filepath = Path(concatenated_results_path) / 'test' / 'concat_data.csv'
    train_filepath_copied_genes = Path(concatenated_results_path) / 'train' / 'concat_data_copied.csv'
    val_filepath_copied_genes = Path(concatenated_results_path) / 'val' / 'concat_data_copied.csv'
    test_filepath_copied_genes = Path(concatenated_results_path) / 'test' / 'concat_data_copied.csv'

    if not os.path.exists(data_path):
        print("%s not existing." % data_path)
        exit()
    # if not os.path.exists(concatenated_results_path):
    #     print("%s not existing." % concatenated_results_path)
    #     exit()
    # if not os.path.exists(train_filepath):
    #     print("%s not existing." % train_filepath)
    #     exit()
    # if not os.path.exists(val_filepath):
    #     print("%s not existing." % val_filepath)
    #     exit()
    # if not os.path.exists(test_filepath):
    #     print("%s not existing." % test_filepath)
    #     exit()
    # if not os.path.exists(train_filepath_copied_genes):
    #     print("%s not existing." % train_filepath_copied_genes)
    #     exit()
    # if not os.path.exists(val_filepath_copied_genes):
    #     print("%s not existing." % val_filepath_copied_genes)
    #     exit()
    # if not os.path.exists(test_filepath_copied_genes):
    #     print("%s not existing." % test_filepath_copied_genes)
    #     exit()

    if args.n_principal_components is not None:
        params['pca']['n_components'] = args.n_principal_components

    if args.plot_explained_variance:
        if not args.n_principal_components:
            print('error: missing argument <n_principal_components>.')
            exit()
        utils.plot_explained_variance_pca(params, train_filepath, val_filepath, test_filepath)

    else:
        if args.classification_method == 'linearsvc' or args.classification_method == 'sgd':
            # if not args.n_principal_components:
            #     print('error: missing argument <n_principal_components>.')
            #     exit()
            shallow_classification.shallow_classifier(args, params, train_filepath, val_filepath, test_filepath, data_path)

        elif args.classification_method == 'pca_nn':
            if not args.n_principal_components:
                print('error: missing argument <n_principal_components>.')
                exit()
            nn_classification.pca_nn_classifier(args, params, train_filepath, val_filepath, test_filepath, data_path)

        elif args.classification_method == 'nn':
            nn_classification.nn_classifier(args, params, train_filepath_copied_genes, val_filepath_copied_genes,
                                            test_filepath_copied_genes)


if __name__ == '__main__':
    main()
