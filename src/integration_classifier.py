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
    parser.add_argument('--classification_method',
                        help='Classification method',
                        choices=['linearsvc', 'sgdclassifier', 'nn', 'pca_nn'],
                        required=False,
                        type=str)
    parser.add_argument('--balancing',
                        help='Class balancing method',
                        choices=['random_upsampling', 'smoteenn', 'smote', 'weights'],
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

    data_path = paths.integration_classification_data_dir

    if not os.path.exists(data_path):
        print("%s not existing." % data_path)
        exit()

    if args.classification_method == 'linearsvc' or args.classification_method == 'sgdclassifier':
        shallow_classification.shallow_classifier(args, params, data_path, n_features_images=params['general']['n_features_images'])

    elif args.classification_method == 'pca_nn':
        nn_classification.pca_nn_classifier(args, params, data_path, n_features_images=params['general']['n_features_images'])

    elif args.classification_method == 'nn':
        nn_classification.nn_classifier(args, params, data_path, n_features_images=params['general']['n_features_images'])


if __name__ == '__main__':
    main()
