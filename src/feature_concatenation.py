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

    if not os.path.exists(args.cfg):
        print(f"{args.cfg} not found")
        exit(-1)

    tile_features_train_dir = paths.extracted_features_train
    tile_features_test_dir = paths.extracted_features_test
    tile_features_val_dir = paths.extracted_features_val
    gene_features_train_dir = paths.svm_t_rfe_selected_features_train
    gene_features_test_dir = paths.svm_t_rfe_selected_features_test
    gene_features_val_dir = paths.svm_t_rfe_selected_features_val

    num_principal_components = params['general']['n_components']
    gene_copy_ratio = params['general']['gene_copy_ratio']

    if num_principal_components is not None:

        path_to_save_pca = Path(paths.concatenated_results_dir) / f'pca{args.n_principal_components}'
        if not os.path.exists(path_to_save_pca):
            os.makedirs(path_to_save_pca)

        n_components = num_principal_components if num_principal_components > 0 else None

        # concatenate data with PCA:
        scaler, ipca, scaler_concatenated = concatenate_features.concatenate_pca(
            lookup_dir_tiles=tile_features_train_dir,
            lookup_dir_genes=gene_features_train_dir,
            path_to_save=path_to_save_pca,
            dataset_name='train',
            n_components=n_components)
        concatenate_features.concatenate_pca(lookup_dir_tiles=tile_features_val_dir,
                                             lookup_dir_genes=gene_features_val_dir,
                                             path_to_save=path_to_save_pca,
                                             dataset_name='val',
                                             scaler=scaler,
                                             ipca=ipca,
                                             scaler_concatenated=scaler_concatenated)
        concatenate_features.concatenate_pca(lookup_dir_tiles=tile_features_test_dir,
                                             lookup_dir_genes=gene_features_test_dir,
                                             path_to_save=path_to_save_pca,
                                             dataset_name='test',
                                             scaler=scaler,
                                             ipca=ipca,
                                             scaler_concatenated=scaler_concatenated)

        if args.plot_explained_variance:
            plots.plot_explained_variance(ipca.explained_variance_ratio_, path_to_save_pca, n_components)
            np.savetxt(Path(path_to_save_pca) / 'pca_components.csv', ipca.components_, delimiter=',', fmt='%s')

    if gene_copy_ratio is not None:

        path_to_save_copied_genes = paths.concatenated_results_dir / f'copyratio{gene_copy_ratio}'
        if not os.path.exists(path_to_save_copied_genes):
            os.makedirs(path_to_save_copied_genes)

        if gene_copy_ratio <= 0:
            print(f'error: invalid configuration <gene_copy_ratio>: {gene_copy_ratio}')

        # concatenate data with repeating genes to match tiles dimensionality (no scaling):
        scaler = concatenate_features.concatenate_copy_genes(lookup_dir_tiles=tile_features_train_dir,
                                                    lookup_dir_genes=gene_features_train_dir,
                                                    path_to_save=path_to_save_copied_genes,
                                                    dataset_name='train',
                                                    gene_copy_ratio=gene_copy_ratio)
        concatenate_features.concatenate_copy_genes(lookup_dir_tiles=tile_features_val_dir,
                                                    lookup_dir_genes=gene_features_val_dir,
                                                    path_to_save=path_to_save_copied_genes,
                                                    dataset_name='val',
                                                    gene_copy_ratio=gene_copy_ratio,
                                                    scaler=scaler)
        concatenate_features.concatenate_copy_genes(lookup_dir_tiles=tile_features_test_dir,
                                                    lookup_dir_genes=gene_features_test_dir,
                                                    path_to_save=path_to_save_copied_genes,
                                                    dataset_name='test',
                                                    gene_copy_ratio=gene_copy_ratio,
                                                    scaler=scaler)


if __name__ == '__main__':
    main()
