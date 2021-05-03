import argparse
import os
from pathlib import Path
from config import paths
from src.common import concatenate_features


def main():
    """
       Description: Concatenate gene features (possibly copied according to specified ratio) with tile features and save them on files.
    """
    tile_features_train_dir = paths.extracted_features_train
    tile_features_test_dir = paths.extracted_features_test
    tile_features_val_dir = paths.extracted_features_val
    gene_features_train_dir = paths.svm_t_rfe_selected_features_train
    gene_features_test_dir = paths.svm_t_rfe_selected_features_test
    gene_features_val_dir = paths.svm_t_rfe_selected_features_val

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

    # concatenate data with and without repeating genes to match tiles dimensionality:

    print('----------------------------')
    print(">> Concatenating train data:")
    print('----------------------------')
    concatenate_features.concatenate(tile_features_train_dir, gene_features_train_dir, path_to_save_train)
    print('>> With repeated gene features:')
    print('-------------------------------')
    concatenate_features.concatenate(tile_features_train_dir, gene_features_train_dir, path_to_save_train, gene_copy_ratio=10)

    print('---------------------------------')
    print(">> Concatenating validation data:")
    print('---------------------------------')
    concatenate_features.concatenate(tile_features_val_dir, gene_features_val_dir, path_to_save_val)
    print('>> With repeated gene features:')
    print('-------------------------------')
    concatenate_features.concatenate(tile_features_val_dir, gene_features_val_dir, path_to_save_val, gene_copy_ratio=10)

    print('---------------------------')
    print(">> Concatenating test data:")
    print('---------------------------')
    concatenate_features.concatenate(tile_features_test_dir, gene_features_test_dir, path_to_save_test)
    print('>> With repeated gene features:')
    print('-------------------------------')
    concatenate_features.concatenate(tile_features_test_dir, gene_features_test_dir, path_to_save_test, gene_copy_ratio=10)


if __name__ == '__main__':
    main()