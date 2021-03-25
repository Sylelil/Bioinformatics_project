import os
import numpy as np
import pandas as pd
from pathlib import Path


def get_tile_data(lookup_dir):
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


def get_gene_data(lookup_dir):
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


def get_split_caseids():
    """
        Get train and test splits of caseids saved in 'assets\train_test_split' folder.
        Returns:
            train_caseids: list of caseids of train split.
            test_caseids: list of caseids of test split.
    """
    lookup_dir = Path('..') / '..' / 'assets' / 'data_splits'
    file_path_train = os.path.join(lookup_dir, 'train_caseids.npy')
    file_path_val = os.path.join(lookup_dir, 'val_caseids.npy')
    file_path_test = os.path.join(lookup_dir, 'test_caseids.npy')
    train_caseids = np.load(file_path_train)
    val_caseids = np.load(file_path_val)
    test_caseids = np.load(file_path_test)
    return train_caseids, val_caseids, test_caseids


def get_split_data(lookup_dir):
    """
       Split data according to splits saved in 'assets\train_test_split' folder.
       Args:
            lookup_dir: lookup directory with data to be split.
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: train, validation and test subsets of data and labels
    """
    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []

    # get splitted caseids
    train_caseids, val_caseids, test_caseids = get_split_caseids()

    # get data with caseid and label
    for np_file in os.listdir(lookup_dir):
        file_path = os.path.join(lookup_dir, np_file)
        data = np.load(file_path)
        filename = os.path.splitext(np_file)[0]
        caseid = filename[:-2]
        label = filename[-1]

        # add data to corresponding split
        if caseid in train_caseids:
            X_train.append(data)
            y_train.append(label)
        elif caseid in val_caseids:
            X_val.append(data)
            y_val.append(label)
        elif caseid in test_caseids:
            X_test.append(data)
            y_test.append(label)
        else:
            print(f"error: caseid {caseid} not found in splits.")
            exit()

    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)
