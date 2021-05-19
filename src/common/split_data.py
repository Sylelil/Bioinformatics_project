import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.genes import utils
from src.images import slide_info
import numpy as np


def get_images_split_data(lookup_dir, val_data=True):
    """
        Description: Retrieve images split data from lookup folder.
        :param lookup_dir: lookup directory with split data.
        :param val_data: either or not to return validation split (default: True)
        :returns: X_train, X_val, X_test, y_train, y_val, y_test: datasets and labels
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
        Description: Retrieve genes split data from lookup folder.
        :param lookup_dir: lookup directory with split data.
        :param val_data: either or not to return validation split (default: True)
        :returns: X_train, (X_val,) X_test, y_train, (y_val,) y_test: datasets and labels
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

    X_train, y_train = utils.read_genes_from_folder(train_filepath)
    X_val, y_val = utils.read_genes_from_folder(val_filepath)
    X_test, y_test = utils.read_genes_from_folder(test_filepath)

    if val_data:
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train_val = X_train.append(X_val, sort=False)
        y_train.extend(y_val)
        return X_train_val, X_test, y_train, y_test


def split_filenames(test_size, save_dir, lookup_dir=None, filenames_arg=None, labels_arg=None, nametrain='train', nametest='test'):
    """
        Description: Split filenames into random train, test subsets according to specified sizes with stratification.
        :param test_size: float or int size of test subset.
        :param save_dir: directory where lists of split filenames will be saved.
        :param lookup_dir: lookup directory with data filenames to be split.
        :param filenames_arg: list of train+val filenames to be split.
        :param labels_arg: list of train+val labels to be split.
        :param nametrain: name of train split.
        :param nametest: name of test split.
        :returns: filenames_train, filenames_test, labels_train, labels_test: split filenames lists and split labels lists
    """
    filenames = []
    labels = []
    if lookup_dir is not None:
        print('>> Reading filenames...')
        # read case ids and labels of all samples
        for np_file in tqdm(os.listdir(lookup_dir)):
            filename = os.path.splitext(np_file)[0]
            label = filename[-1]
            filenames.append(filename)
            labels.append(label)
    elif filenames_arg is not None and labels_arg is not None:
        filenames.extend(filenames_arg)
        labels.extend(labels_arg)
    else:
        print('Error: missing argument lookup_dir or filenames_arg and labels_arg.')
        exit()

    print(f'>> Splitting filenames in {nametrain} and {nametest}...')
    # split train and test
    filenames_train, filenames_test, labels_train, labels_test = train_test_split(filenames, labels,
                                                                                  test_size=test_size, stratify=labels,
                                                                                  random_state=42)
    # save in files
    print(f'>> Saving splits in {save_dir} directory...')
    fname_train = nametrain + '_filenames.npy'
    fname_test = nametest + '_filenames.npy'
    np.save(os.path.join(save_dir, fname_train), filenames_train)
    np.save(os.path.join(save_dir, fname_test), filenames_test)
    print('>> Done')

    return filenames_train, filenames_test, labels_train, labels_test


def split_into_folders(lookup_dir, filenames_train, filenames_val, filenames_test, path_to_save):
    """
        Description: Copy data into folders according to filenames splits.
        :param lookup_dir: lookup directory with data to be split.
        :param filenames_train: list of train filenames.
        :param filenames_val: list of val filenames.
        :param filenames_test: list of test filenames.
        :param path_to_save: directory where data will be saved.
    """
    traindir = Path(path_to_save) / 'train'
    valdir = Path(path_to_save) / 'val'
    testdir = Path(path_to_save) / 'test'
    print(f'>> Copying train, validation and test data in subfolders in {path_to_save}')
    skipped_filenames = []
    n_train = 0
    n_test = 0
    n_val = 0
    for file in tqdm(os.listdir(lookup_dir)):
        file_path = os.path.join(lookup_dir, file)
        if os.path.isdir(file_path):
            for slide_file in os.listdir(file_path):
                if slide_file.endswith('.svs'):
                    filename = os.path.splitext(slide_file)[0]
                    file_path = os.path.join(file_path, slide_file)

                    if filename in filenames_train:
                        dest_path = Path(traindir) / str(slide_file)
                        n_train += 1
                    elif filename in filenames_test:
                        dest_path = Path(testdir) / str(slide_file)
                        n_test += 1
                    elif filename in filenames_val:
                        dest_path = Path(valdir) / str(slide_file)
                        n_val += 1
                    else:
                        if filename not in skipped_filenames:
                            skipped_filenames.append(filename)
                        continue
                    shutil.copyfile(file_path, dest_path)
        else:
            filename = os.path.splitext(file)[0]
            if filename in filenames_train:
                dest_path = Path(traindir) / file
                n_train += 1
            elif filename in filenames_test:
                dest_path = Path(testdir) / file
                n_test += 1
            elif filename in filenames_val:
                dest_path = Path(valdir) / file
                n_val += 1
            else:
                if filename not in skipped_filenames:
                    skipped_filenames.append(filename)
                continue
            shutil.copyfile(file_path, dest_path)

    print(f'{len(skipped_filenames)} skipped filename (not in splits): {skipped_filenames}')
    print(f'{n_train} train files copied.')
    print(f'{n_val} validation files copied.')
    print(f'{n_test} test files copied.')
    print('>> Done')
