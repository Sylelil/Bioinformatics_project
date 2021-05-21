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

    args = parser.parse_args()
    return args


def main():
    """
       Description: Concatenate gene features (possibly copied according to specified ratio) with tile features and save them on files.
    """
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    if not os.path.exists(args.cfg):
        print(f"{args.cfg} not found")
        exit(-1)
    params = utils.read_config_file(args.cfg)

    tile_features_train_dir = paths.extracted_features_train
    tile_features_test_dir = paths.extracted_features_test
    tile_features_val_dir = paths.extracted_features_val
    gene_features_train_dir = paths.svm_t_rfe_selected_features_train
    gene_features_test_dir = paths.svm_t_rfe_selected_features_test
    gene_features_val_dir = paths.svm_t_rfe_selected_features_val

    num_principal_components = params['general']['num_principal_components']
    use_features_images_only = params['general']['use_features_images_only']

    if use_features_images_only:

        if num_principal_components is not None:
            path_to_save = Path(paths.concatenated_results_dir) / 'images' / f'pca{num_principal_components}'
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            n_components = num_principal_components if num_principal_components > 0 else None

            scaler, ipca = concatenate_features.save_features_images_only(
                lookup_dir_tiles_train=tile_features_train_dir,
                lookup_dir_tiles_val=tile_features_val_dir,
                path_to_save=path_to_save,
                dataset_name='train',
                n_components=n_components,
                with_ipca=True)
            concatenate_features.save_features_images_only(lookup_dir_tiles_test=tile_features_test_dir,
                                                           path_to_save=path_to_save,
                                                           dataset_name='test',
                                                           n_components=n_components,
                                                           scaler=scaler,
                                                           ipca=ipca,
                                                           with_ipca=True)
            plots.plot_explained_variance(ipca.explained_variance_ratio_, path_to_save, n_components)
            np.savetxt(Path(path_to_save) / 'pca_components.csv', ipca.components_, delimiter=',', fmt='%s')

        else:
            path_to_save = Path(paths.concatenated_results_dir) / 'images' / 'all'
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            scaler, _ = concatenate_features.save_features_images_only(lookup_dir_tiles_train=tile_features_train_dir,
                                                                       lookup_dir_tiles_val=tile_features_val_dir,
                                                                       path_to_save=path_to_save,
                                                                       dataset_name='train',
                                                                       with_ipca=False)
            concatenate_features.save_features_images_only(lookup_dir_tiles_test=tile_features_test_dir,
                                                           path_to_save=path_to_save,
                                                           dataset_name='test',
                                                           scaler=scaler,
                                                           with_ipca=False)

    else:
        if num_principal_components is not None:
            path_to_save = Path(
                paths.concatenated_results_dir) / 'concatenated' / f'pca{num_principal_components}'
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            n_components = num_principal_components if num_principal_components > 0 else None

            # concatenate data with PCA:
            scaler, ipca = concatenate_features.concatenate(
                lookup_dir_tiles_train=tile_features_train_dir,
                lookup_dir_tiles_val=tile_features_val_dir,
                lookup_dir_genes_train=gene_features_train_dir,
                lookup_dir_genes_val=gene_features_val_dir,
                path_to_save=path_to_save,
                dataset_name='train',
                n_components=n_components,
                with_ipca=True)
            concatenate_features.concatenate(lookup_dir_tiles_test=tile_features_test_dir,
                                             lookup_dir_genes_test=gene_features_test_dir,
                                             path_to_save=path_to_save,
                                             dataset_name='test',
                                             scaler=scaler,
                                             ipca=ipca,
                                             with_ipca=True)

            plots.plot_explained_variance(ipca.explained_variance_ratio_, path_to_save, n_components)
            np.savetxt(Path(path_to_save) / 'pca_components.csv', ipca.components_, delimiter=',', fmt='%s')

        else:
            path_to_save = paths.concatenated_results_dir / 'concatenated' / 'all'
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)

            # concatenate data with repeating genes to match tiles dimensionality (no scaling):
            scaler, _ = concatenate_features.concatenate(lookup_dir_tiles_train=tile_features_train_dir,
                                                         lookup_dir_tiles_val=tile_features_val_dir,
                                                         lookup_dir_genes_train=gene_features_train_dir,
                                                         lookup_dir_genes_val=gene_features_val_dir,
                                                         path_to_save=path_to_save,
                                                         dataset_name='train',
                                                         with_ipca=False)
            concatenate_features.concatenate(lookup_dir_tiles_test=tile_features_test_dir,
                                             lookup_dir_genes_test=gene_features_test_dir,
                                             path_to_save=path_to_save,
                                             dataset_name='test',
                                             scaler=scaler,
                                             with_ipca=False)


if __name__ == '__main__':
    main()
