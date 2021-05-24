import argparse
import os
from pathlib import Path
from config import paths
from common import classification_report_utils
from integration import utils
from integration.classification_methods import nn_classification, shallow_classification


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
                        choices=['linearsvc', 'sgdclassifier', 'nn', 'pcann'],
                        required=False,
                        type=str)
    parser.add_argument('--plot_final_results',
                        help='Either to plot final results or not',
                        required=False,
                        action='store_true')

    args = parser.parse_args()
    return args


def main():
    """
       Description: Train and test classifiers on concatenated features.
    """
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    if not os.path.exists(args.cfg):
        print(f"{args.cfg} not found")
        exit(-1)
    params = utils.read_config_file(args.cfg)

    num_principal_components = params['general']['num_principal_components']
    use_features_images_only = params['general']['use_features_images_only']

    if use_features_images_only:
        data_folder = Path(paths.concatenated_results_dir) / 'images'
    else:
        data_folder = Path(paths.concatenated_results_dir) / 'concatenated'

    if args.classification_method == 'linearsvc' or args.classification_method == 'sgdclassifier':
        data_path = Path(data_folder) / f"pca{num_principal_components}"
        if not os.path.exists(data_path):
            print("%s not existing." % data_path)
            exit(-1)
        shallow_classification.shallow_classifier(args, params, data_path)

    elif args.classification_method == 'pcann':
        data_path = Path(data_folder) / f"pca{num_principal_components}"
        if not os.path.exists(data_path):
            print("%s not existing." % data_path)
            exit(-1)
        nn_classification.nn_classifier(args, params['pcann'], params, data_path)

    elif args.classification_method == 'nn':
        data_path = Path(data_folder) / f'all'
        if not os.path.exists(data_path):
            print("%s not existing." % data_path)
            exit(-1)
        nn_classification.nn_classifier(args, params['nn'], params, data_path)

    elif args.plot_final_results:
        results_path = paths.integration_classification_results_dir
        if not os.path.exists(results_path):
            print("%s not existing." % results_path)
            exit()
        classification_report_utils.generate_final_classification_plots(results_path)


if __name__ == '__main__':
    main()
