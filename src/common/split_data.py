import argparse
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter


def save_split_data_files(X_train, X_test, y_train, y_test, path_to_save):
    """
        Description: Save split datasets in separate files.
        :param X_train: training dataset.
        :param X_test: test dataset.
        :param y_train: numpy array of training labels.
        :param y_test: numpy array of test labels.
        :param path_to_save: Path of the directory where files will be saved.
    """
    if not os.path.exists(path_to_save):
        print("%s not existing." % path_to_save)
        exit()
    print(f'>> Saving split data and labels in {path_to_save} directory...')
    with open(os.path.join(path_to_save, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(path_to_save, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(path_to_save, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(path_to_save, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    print('>> Done')


def read_split_data_from_files(lookup_dir):
    """
        Description: Retrieve split datasets from files.
        :param lookup_dir: lookup directory with data.
        :return: X_train, X_test, y_train, y_test: train and test datasets and labels
    """
    if not os.path.exists(lookup_dir):
        print("%s not existing." % lookup_dir)
        exit()
    print('>> Retrieving split data from files...')
    with open(os.path.join(lookup_dir, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(lookup_dir, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(lookup_dir, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(lookup_dir, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    print('>> Done')
    return X_train, X_test, y_train, y_test


def __get_split_caseids(lookup_dir):
    """
        Description: Private function. Get train and test splits of caseids saved in 'assets\train_test_split' folder.
        :return: numpy arrays of caseids of train and test split.
    """
    print('>> Retrieving split caseids...')
    file_path_train = os.path.join(lookup_dir, 'train_caseids.npy')
    file_path_test = os.path.join(lookup_dir, 'test_caseids.npy')
    train_caseids = np.load(file_path_train)
    test_caseids = np.load(file_path_test)
    print('>> Done')
    return train_caseids, test_caseids


def get_split_data(lookup_dir, splits_dir, path_to_save=None):
    """
        Description: Retrieve split data according to lists of caseids saved in 'assets\train_test_split' folder.
        :param lookup_dir: lookup directory with data to be split.
        :param path_to_save: directory for saving data (default=None).
        :return: X_train, X_test, y_train, y_test: train and test datasets and labels
    """

    if not os.path.exists(lookup_dir):
        print("%s not existing." % lookup_dir)
        exit()

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # get splitted caseids
    train_caseids, test_caseids = __get_split_caseids(splits_dir)

    # get data with caseid and label
    print('>> Retrieving data based on caseid splits...')
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
        elif caseid in test_caseids:
            X_test.append(data)
            y_test.append(label)
        else:
            print(f"error: caseid {caseid} not found in splits.")
            exit()
    print('>> Done')

    if path_to_save:
        save_split_data_files(X_train, X_test, y_train, y_test, path_to_save)

    return X_train, X_test, y_train, y_test


def __split_caseids(lookup_dir, test_size, path_to_save):
    """
        Description: Private function. Split caseids into random train, test subsets according to specified sizes with stratification.
        Save splitted caseids into three files in 'assets\train_test_split' folder.
        :param lookup_dir: lookup directory with data to be split.
        :param test_size: float or int size of test subset.
    """

    print('>> Reading data...')
    # read case ids and labels of all samples
    caseids = []
    labels = []
    for np_file in os.listdir(lookup_dir):
        filename = os.path.splitext(np_file)[0]
        caseids.append(filename)
        labels.append(filename[-1])

    '''
    # check if there are duplicates
    duplicates = [k for k, v in Counter(caseids).items() if v > 1]
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated caseids: {duplicates}")
        exit()
        '''

    print('>> Splitting caseids...')
    # split train and test
    caseids_train, caseids_test, _, _ = train_test_split(caseids, labels, test_size=test_size,
                                                                          stratify=labels, random_state=42)

    # save in files
    print(f'>> Saving splits in {path_to_save} directory...')
    np.save(os.path.join(path_to_save, 'train_caseids.npy'), caseids_train)
    np.save(os.path.join(path_to_save, 'test_caseids.npy'), caseids_test)
    print('>> Done')

