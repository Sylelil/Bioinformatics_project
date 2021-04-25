import argparse
import shutil

import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter

from tqdm import tqdm

from config import paths
from src.genes import methods
from src.images import slide_info


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


def get_images_split_data(lookup_dir, val_data=True):
    """
        Description: Retrieve split data from lookup folder.
        :param lookup_dir: lookup directory with split data.
        :return: X_train, X_val, X_test, y_train, y_val, y_test: datasets and labels
    """
    train_filepath = Path(lookup_dir) / 'train'
    val_filepath = Path(lookup_dir) / 'val'
    test_filepath = Path(lookup_dir) / 'test'

    if not os.path.exists(train_filepath):
        print("%s not existing." % train_filepath)
        exit()
    if not os.path.exists(val_filepath):
        print("%s not existing." % val_filepath)
        exit()
    if not os.path.exists(test_filepath):
        print("%s not existing." % test_filepath)
        exit()

    X_train, y_train = slide_info.read_slides_info_from_folder(train_filepath)
    X_val, y_val = slide_info.read_slides_info_from_folder(val_filepath)
    X_test, y_test = slide_info.read_slides_info_from_folder(test_filepath)

    if val_data:
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train.extend(X_val)
        y_train.extend(y_val)
        return X_train, X_test, y_train, y_test


def get_genes_split_data(lookup_dir, val_data=True):
    """
        Description: Retrieve split data from lookup folder.
        :param lookup_dir: lookup directory with split data.
        :return: X_train, X_val, X_test, y_train, y_val, y_test: datasets and labels
    """
    train_filepath = Path(lookup_dir) / 'train'
    val_filepath = Path(lookup_dir) / 'val'
    test_filepath = Path(lookup_dir) / 'test'

    if not os.path.exists(train_filepath):
        print("%s not existing." % train_filepath)
        exit()
    if not os.path.exists(val_filepath):
        print("%s not existing." % val_filepath)
        exit()
    if not os.path.exists(test_filepath):
        print("%s not existing." % test_filepath)
        exit()

    X_train, y_train = methods.read_genes_from_folder(train_filepath)
    X_val, y_val = methods.read_genes_from_folder(val_filepath)
    X_test, y_test = methods.read_genes_from_folder(test_filepath)

    if val_data:
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train_val = X_train.append(X_val, sort=False)
        y_train.extend(y_val)
        return X_train_val, X_test, y_train, y_test


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
        cid=''
        file_path = os.path.join(lookup_dir, file)
        if os.path.isdir(file_path):
            for slide_file in os.listdir(file_path):
                if slide_file.endswith('.svs'):
                    filename = os.path.splitext(slide_file)[0]
                    cid = filename[:-2]
                    file_path = os.path.join(file_path, slide_file)

                    if cid in caseids_train:
                        dest_path = Path(traindir) / str(slide_file)
                        n_train += 1
                    elif cid in caseids_test:
                        dest_path = Path(testdir) / str(slide_file)
                        n_test += 1
                    elif cid in caseids_val:
                        dest_path = Path(valdir) / str(slide_file)
                        n_val += 1
                    else:
                        if cid not in skipped_caseids:
                            skipped_caseids.append(cid)
                        continue
                    shutil.copyfile(file_path, dest_path)
        else:
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
