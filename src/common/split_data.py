import argparse
import shutil

import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter

from tqdm import tqdm


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


def get_images_split_data(slides_info, splits_dir, path_to_save=None):
    """
        Description: Retrieve split data according to lists of caseids saved in splits_dir.
        :param lookup_dir: lookup directory with data to be split.
        :param splits_dir: directory with lists of splits.
        :param path_to_save: directory for saving data (default=None).
        :return: X_train, X_test, y_train, y_test: train and test datasets (lists)
    """

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # get split caseids
    train_caseids, test_caseids = __get_split_caseids(splits_dir)

    # get data with caseid and label
    print('>> Retrieving data based on caseid splits...')
    for slide in slides_info:
        filename = slide['slide_name']
        label = filename[-1]

        # add data to corresponding split
        if filename in train_caseids:
            X_train.append(slide)
            y_train.append(label)
        elif filename in test_caseids:
            X_test.append(slide)
            y_test.append(label)
        else:
            print(f"warning: caseid {filename} not found in splits.")
            # exit()
    print('>> Done')

    if path_to_save:
        save_split_data_files(X_train, X_test, y_train, y_test, path_to_save)

    return X_train, X_test, y_train, y_test


def get_genes_split_data(df, splits_dir, path_to_save=None):
    """
        Description: Retrieve split data according to lists of caseids saved in splits_dir.
        :param lookup_dir: lookup directory with data to be split.
        :param path_to_save: directory for saving data (default=None).
        :return: X_train, X_test, y_train, y_test: train and test datasets and labels
    """

    # get split caseids
    train_caseids, test_caseids = __get_split_caseids(splits_dir)

    # get data with caseid and label
    X_train = df.loc[train_caseids]
    X_test = df.loc[test_caseids]
    y_train = np.array([int(x[-1:]) for x in train_caseids])
    y_test = np.array([int(x[-1:]) for x in test_caseids])
    print('>> Done')

    if path_to_save:
        save_split_data_files(X_train, X_test, y_train, y_test, path_to_save)

    return X_train, X_test, y_train, y_test


def split_caseids(test_size, save_dir, lookup_dir=None, caseids_arg=None, labels_arg=None, nametrain='train', nametest='test'):
    """
        Description: Private function. Split caseids into random train, test subsets according to specified sizes with stratification.
        Save splitted caseids into three files in 'assets\train_test_split' folder.
        :param lookup_dir: lookup directory with data to be split.
        :param test_size: float or int size of test subset.
    """
    if lookup_dir is not None:
        print('>> Reading filenames...')
        # read case ids and labels of all samples
        filenames = []
        caseids_dict = {}
        duplicated = []
        for np_file in tqdm(os.listdir(lookup_dir)):
            filename = os.path.splitext(np_file)[0]
            cid = filename[:-2]
            lab = filename[-1]
            # check if there are duplicates
            if cid in caseids_dict:
                duplicated.append(cid)
            else:
                caseids_dict[cid] = [cid, lab]

        # remove duplicates
        for dup in duplicated:
            caseids_dict.pop(dup)

        print(f'{len(duplicated)} duplicated caseids found and skipped: {duplicated}')

        caseids_labels = np.asarray(list(caseids_dict.values()))
        caseids = caseids_labels[:, 0]
        labels = caseids_labels[:, 1]

    elif caseids_arg is not None and labels_arg is not None:
        caseids = caseids_arg
        labels = labels_arg
    else:
        print('Error: missing argument lookup_dir or caseids and labels.')
        exit()

    print(f'>> Splitting caseids in {nametrain} and {nametest}...')
    # split train and test
    caseids_train, caseids_test, labels_train, labels_test = train_test_split(caseids, labels,
                                                         test_size=test_size, stratify=labels, random_state=42)

    # save in files
    print(f'>> Saving splits in {save_dir} directory...')
    filename_train = nametrain + '_caseids.npy'
    filename_test = nametest + '_caseids.npy'
    np.save(os.path.join(save_dir, filename_train), caseids_train)
    np.save(os.path.join(save_dir, filename_test), caseids_test)
    print('>> Done')

    return caseids_train, caseids_test, labels_train, labels_test


def split_into_folders(lookup_dir, caseids_train, caseids_val, caseids_test, path_to_save):
    traindir = Path(path_to_save) / 'train'
    valdir = Path(path_to_save) / 'val'
    testdir = Path(path_to_save) / 'test'
    print(f'>> Copying train, validation and test data in subfolders in {path_to_save}')
    skipped_caseids = []
    n_train = 0
    n_test = 0
    n_val = 0
    for file in tqdm(os.listdir(lookup_dir)):
        file_path = os.path.join(lookup_dir, file)
        filename = os.path.splitext(file)[0]
        cid = filename[:-2]
        if cid in caseids_train:
            dest_path = Path(traindir) / file
            n_train += 1
        elif cid in caseids_test:
            dest_path = Path(testdir) / file
            n_test += 1
        elif cid in caseids_val:
            dest_path = Path(valdir) / file
            n_val += 1
        else:
            if cid not in skipped_caseids:
                skipped_caseids.append(cid)
            continue
        shutil.copyfile(file_path, dest_path)
    print(f'{len(skipped_caseids)} skipped caseid (not in splits): {skipped_caseids}')
    print(f'{n_train} train files copied.')
    print(f'{n_val} validation files copied.')
    print(f'{n_test} test files copied.')
    print('>> Done')
