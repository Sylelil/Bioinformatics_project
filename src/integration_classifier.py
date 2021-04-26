import argparse
import os
from pathlib import Path
from config import paths
from src.common import utils
from src.common.integration_classification_methods import nn_classification, shallow_classification
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
    parser.add_argument('--method',
                        help='Classification method',
                        choices=['svc', 'sgd', 'nn', 'pca_nn'],
                        required=True,
                        type=str)
    parser.add_argument('--balancing',
                        help='Class balancing method',
                        choices=['random_upsampling', 'combined', 'smote', 'downsampling', 'weights'],
                        required=False,
                        type=str)
    args = parser.parse_args()
    return args


def main():
    """
       Description: Train and test classifier on concatenated features, with possible preprocessing techniques and class balancing.
    """
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    params = utils.read_config_file(args.cfg)

    concatenated_results_path = paths.concatenated_results_dir
    train_filepath = Path(concatenated_results_path) / 'train' / 'concat_data.csv'
    val_filepath = Path(concatenated_results_path) / 'val' / 'concat_data.csv'
    test_filepath = Path(concatenated_results_path) / 'test' / 'concat_data.csv'
    train_filepath_copied_genes = Path(concatenated_results_path) / 'train' / 'concat_data_copied.csv'
    val_filepath_copied_genes = Path(concatenated_results_path) / 'val' / 'concat_data_copied.csv'
    test_filepath_copied_genes = Path(concatenated_results_path) / 'test' / 'concat_data_copied.csv'

    if not os.path.exists(concatenated_results_path):
        print("%s not existing." % concatenated_results_path)
        exit()
    if not os.path.exists(train_filepath):
        print("%s not existing." % train_filepath)
        exit()
    if not os.path.exists(val_filepath):
        print("%s not existing." % val_filepath)
        exit()
    if not os.path.exists(test_filepath):
        print("%s not existing." % test_filepath)
        exit()
    if not os.path.exists(train_filepath_copied_genes):
        print("%s not existing." % train_filepath_copied_genes)
        exit()
    if not os.path.exists(val_filepath_copied_genes):
        print("%s not existing." % val_filepath_copied_genes)
        exit()
    if not os.path.exists(test_filepath_copied_genes):
        print("%s not existing." % test_filepath_copied_genes)
        exit()

    if args.method == 'svc' or args.method == 'sgd':
        shallow_classification.shallow_classifier(args, params, train_filepath, val_filepath, test_filepath)
    elif args.method == 'pca_nn':
        nn_classification.pca_nn_classifier(args, params, train_filepath, val_filepath, test_filepath)
    elif args.method == 'nn':
        nn_classification.nn_classifier(args, params, train_filepath_copied_genes, val_filepath_copied_genes, test_filepath_copied_genes)



if __name__ == '__main__':
    main()