import argparse
import os
from pathlib import Path
from config import paths
from src.common import concatenate_features, plots
from src.integration import utils
import numpy as np


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
    parser.add_argument('--gene_copy_ratio',
                        help='Gene copy ratio',
                        required=False,
                        default=1,
                        type=int)
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
       Description: Concatenate gene features (possibly copied according to specified ratio) with tile features and save them on files.
    """
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    params = utils.read_config_file(args.cfg)

    # tile_features_train_dir = paths.extracted_features_train
    # tile_features_test_dir = paths.extracted_features_test
    # tile_features_val_dir = paths.extracted_features_val
    # gene_features_train_dir = paths.svm_t_rfe_selected_features_train
    # gene_features_test_dir = paths.svm_t_rfe_selected_features_test
    # gene_features_val_dir = paths.svm_t_rfe_selected_features_val

    tile_features_train_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'split_data' / 'images' / 'train'
    tile_features_test_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'split_data' / 'images' / 'test'
    tile_features_val_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'split_data' / 'images' / 'val'
    gene_features_train_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'split_data' / 'genes' / 'train'
    gene_features_test_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'split_data' / 'genes' / 'test'
    gene_features_val_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'split_data' / 'genes' / 'val'

    if args.n_principal_components is not None:
        path_to_save = Path(paths.concatenated_results_dir) / f'pca_{args.n_principal_components}'
    else:
        path_to_save = paths.concatenated_results_dir
    path_to_save_train = Path(path_to_save) / 'train'
    path_to_save_test = Path(path_to_save) / 'test'
    path_to_save_val = Path(path_to_save) / 'val'

    if not os.path.exists(tile_features_train_dir):
        print("%s not existing." % tile_features_train_dir)
        exit()
    if not os.path.exists(tile_features_test_dir):
        print("%s not existing." % tile_features_test_dir)
        exit()
    if not os.path.exists(tile_features_val_dir):
        print("%s not existing." % tile_features_val_dir)
        exit()
    if not os.path.exists(gene_features_train_dir):
        print("%s not existing." % gene_features_train_dir)
        exit()
    if not os.path.exists(gene_features_test_dir):
        print("%s not existing." % gene_features_test_dir)
        exit()
    if not os.path.exists(gene_features_val_dir):
        print("%s not existing." % gene_features_val_dir)
        exit()

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(path_to_save_train):
        os.mkdir(path_to_save_train)
    if not os.path.exists(path_to_save_test):
        os.mkdir(path_to_save_test)
    if not os.path.exists(path_to_save_val):
        os.mkdir(path_to_save_val)

    if args.n_principal_components is not None:
        n_components = args.n_principal_components if args.n_principal_components > 0 else None
        # concatenate data with PCA:
        scaler, ipca, scaler_concatenated = concatenate_features.concatenate_pca(lookup_dir_tiles=tile_features_train_dir,
                                                            lookup_dir_genes=gene_features_train_dir,
                                                            path_to_save=path_to_save,
                                                            dataset_name='train',
                                                            n_components=n_components)
        if args.plot_explained_variance:
            plots.plot_explained_variance(ipca.explained_variance_ratio_, path_to_save, n_components)
            np.savetxt(Path(path_to_save) / 'pca_components.csv', ipca.components_, delimiter=',', fmt='%s')
        concatenate_features.concatenate_pca(lookup_dir_tiles=tile_features_val_dir,
                                             lookup_dir_genes=gene_features_val_dir,
                                             path_to_save=path_to_save,
                                             dataset_name='val',
                                             scaler=scaler,
                                             ipca=ipca,
                                             scaler_concatenated=scaler_concatenated)
        concatenate_features.concatenate_pca(lookup_dir_tiles=tile_features_test_dir,
                                             lookup_dir_genes=gene_features_test_dir,
                                             path_to_save=path_to_save,
                                             dataset_name='test',
                                             scaler=scaler,
                                             ipca=ipca,
                                             scaler_concatenated=scaler_concatenated)
    else:
        # concatenate data with and without repeating genes to match tiles dimensionality:
        gene_copy_ratio = args.gene_copy_ratio
        concatenate_features.concatenate(tile_features_train_dir, gene_features_train_dir, path_to_save_train, 'train')
        concatenate_features.concatenate(tile_features_train_dir, gene_features_train_dir, path_to_save_train, 'train',
                                         gene_copy_ratio=gene_copy_ratio)
        concatenate_features.concatenate(tile_features_val_dir, gene_features_val_dir, path_to_save_val, 'val')
        concatenate_features.concatenate(tile_features_val_dir, gene_features_val_dir, path_to_save_val, 'val',
                                         gene_copy_ratio=gene_copy_ratio)
        concatenate_features.concatenate(tile_features_test_dir, gene_features_test_dir, path_to_save_test, 'test')
        concatenate_features.concatenate(tile_features_test_dir, gene_features_test_dir, path_to_save_test, 'test',
                                         gene_copy_ratio=gene_copy_ratio)


if __name__ == '__main__':
    main()
