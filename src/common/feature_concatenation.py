import os
from pathlib import Path

import numpy as np
import pandas as pd


def __get_tile_data(lookup_dir):
    """
       Description: Private function. Read extracted tile features from files.
       :param lookup_dir: Path of the lookup directory.
       :return: Dataframe of features.
    """
    all_tiles_features = []  # list of all tile features
    slide_data = None

    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        filename = os.path.splitext(np_file)[0]
        caseid = filename[:-2]
        label = filename[-1]

        # get list of tile features of a single slide from file
        slide_data = np.load(file_path)
        # shape of slide_data:
        #   [[coordx_1,coordy_1, feat1_1, feat2_1, feat3_1, ...]   # tile 1
        #    [coordx_2,coordy_2, feat1_2, feat2_2, feat3_2, ...]   # tile 2
        #    [coordx_3,coordy_3, feat1_3, feat2_3, feat3_3, ...]   # tile 3
        #    [...]]                                                # tile ...

        # append to each row of dataframe the caseid and label of that slide
        caseid_label_col = [[caseid, label]] * slide_data.shape[0]
        slide_data_caseid_label = np.append(slide_data, caseid_label_col, axis=1)
        # shape of slide_data_caseid_label:
        #   [[coordx_1,coordy_1, feat1_1, feat2_1, feat3_1, ..., caseid, label]   # tile 1
        #    [coordx_2,coordy_2, feat1_2, feat2_2, feat3_2, ..., caseid, label]   # tile 2
        #    [coordx_3,coordy_3, feat1_3, feat2_3, feat3_3, ..., caseid, label]   # tile 3
        #    [...]]                                                               # tile ...

        # add to list of all tile features
        slide_data_list = list(slide_data_caseid_label)
        all_tiles_features.extend(slide_data_list)

    # convert to dataframe
    col_names = ['coord0', 'coord1']
    col_names.extend([f'feat{x}' for x in range(slide_data.shape[1] - 2)])
    col_names.extend(['caseid', 'label'])
    df_tiles_features = pd.DataFrame(all_tiles_features, columns=col_names).set_index('caseid')

    print(df_tiles_features.shape)

    return df_tiles_features


def __get_gene_data(lookup_dir):
    """
       Description: Private function. Read extracted gene features from files.
       :param lookup_dir: Path of the lookup directory.
       :return: Dataframe of features.
    """
    all_features = []  # list of all tile features
    data = None

    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        filename = os.path.splitext(np_file)[0]
        caseid = filename[:-2]
        label = filename[-1]

        # get features of a patient from file
        data = np.load(file_path)
        # shape of data:
        #   [feat1, feat2, feat3, ...]

        # append to each row of dataframe the caseid and label of that slide
        data_caseid_label = np.append(data, [caseid, label])
        # shape of data_caseid_label:
        #   [feat1, feat2, feat3, ..., caseid, label]

        # add to list of all tile features
        data_list = list(data_caseid_label)
        all_features.extend(data_list)

    # convert to dataframe
    col_names = [f'feat{x}' for x in range(data.shape[0])]
    col_names.extend(['caseid', 'label'])
    df_gene_features = pd.DataFrame(all_features, columns=col_names).set_index('caseid')

    print(df_gene_features.shape)

    return df_gene_features


def read_extracted_features():
    """
       Description: Read extracted features from results folders.
       :return: Train and test splits of gene and tile data.
    """
    print(">> Reading features from files...")

    tile_features_train_dir = Path('..') / '..' / 'results' / 'images' / 'extracted_features' / 'training'
    tile_features_test_dir = Path('..') / '..' / 'results' / 'images' / 'extracted_features' / 'test'
    gene_features_train_dir = Path('..') / '..' / 'results' / 'genes' / 'extracted_features' / 'training'
    gene_features_test_dir = Path('..') / '..' / 'results' / 'genes' / 'extracted_features' / 'test'

    if not os.path.exists(Path('..') / '..' / 'results'):
        print("%s not existing." % Path('results'))
        exit()
    if not os.path.exists(tile_features_train_dir):
        print("%s not existing." % tile_features_train_dir)
        exit()
    if not os.path.exists(tile_features_test_dir):
        print("%s not existing." % tile_features_test_dir)
        exit()
    if not os.path.exists(gene_features_train_dir):
        print("%s not existing." % gene_features_train_dir)
        exit()
    if not os.path.exists(gene_features_test_dir):
        print("%s not existing." % gene_features_test_dir)
        exit()

    # get features:
    tile_features_train = __get_tile_data(tile_features_train_dir)
    tile_features_test = __get_tile_data(tile_features_test_dir)
    gene_features_train = __get_gene_data(gene_features_train_dir)
    gene_features_test = __get_gene_data(gene_features_test_dir)

    print(f">> tile_features_train: {tile_features_train.shape}")
    print(f">> tile_features_test: {tile_features_test.shape}")
    print(f">> gene_features_train: {gene_features_train.shape}")
    print(f">> gene_features_test: {gene_features_test.shape}")

    return tile_features_train, tile_features_test, gene_features_train, gene_features_test


def concatenate(tile_features, gene_features, gene_copy_ratio=1):
    """
       Description: Concatenate the features of a tile with the gene features of the corresponding patient.
       :param tile_features: Tile features
       :param gene_features: Gene features
       :param gene_copy_ratio: Ratio by which gene features are copied to match the dimensionality of the tile. Default is 1.
       :return: Array of concatenated features
    """

    if gene_copy_ratio == 1:
        df_genes = gene_features
    else:
        np_gene_features = gene_features.to_numpy()
        # separate caseids and labels from features
        caseids_labels = np_gene_features[:, -2:]
        features = np_gene_features[:, :-2]
        # copy features by ratio
        features_copied = np.tile(features, gene_copy_ratio)
        # re-append caseids and labels to copied features
        np_genes = np.append(features_copied, caseids_labels, axis=1)
        # convert back to dataframe
        col_names = [f'feat{x}' for x in range(features_copied.shape[0])]
        col_names.extend(['caseid', 'label'])
        df_genes = pd.DataFrame(np_genes, columns=col_names).set_index('caseid')

    # perform (inner) join between dataframes based on caseids (indexes)
    #   since there are multiple tile feature lists for a single caseid, and only one gene feature list for a
    #   single caseid, that gene feature list will be duplicated and attached to each tile feature list with that caseid
    concatenated_features = tile_features.merge(df_genes, left_index=True, right_index=True)
    labels = concatenated_features['label'].values

    return concatenated_features, labels
