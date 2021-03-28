import argparse
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def __get_split_caseids():
    """
        Get train and test splits of caseids saved in 'assets\train_test_split' folder.
        Returns:
            train_caseids: list of caseids of train split.
            test_caseids: list of caseids of test split.
    """
    lookup_dir = Path('..') / '..' / 'assets' / 'data_splits'
    file_path_train = os.path.join(lookup_dir, 'train_caseids.npy')
    file_path_test = os.path.join(lookup_dir, 'test_caseids.npy')
    train_caseids = np.load(file_path_train)
    test_caseids = np.load(file_path_test)
    return train_caseids, test_caseids


def get_split_data(lookup_dir):
    """
       Split data according to splits saved in 'assets\train_test_split' folder.
       Args:
            lookup_dir: lookup directory with data to be split.
        Returns:
            X_train, X_test, y_train, y_test: train, validation and test subsets of data and labels
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # get splitted caseids
    train_caseids, test_caseids = __get_split_caseids()

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
        elif caseid in test_caseids:
            X_test.append(data)
            y_test.append(label)
        else:
            print(f"error: caseid {caseid} not found in splits.")
            exit()

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def __split_data(lookup_dir, test_size, val_size):
    """
       Split data into random train, test subsets according to specified sizes with stratification.
       Saves splitted caseids into three files in 'assets\train_test_split' folder.
       Args:
            lookup_dir: lookup directory with data to be split.
            test_size: float or int size of test subset.
    """

    path_to_save = Path('..') / '..' / 'assets' / 'data_splits'

    print('>> Reading data...')
    # read case ids and labels of all samples
    caseids = []
    labels = []
    for np_file in os.listdir(lookup_dir):
        filename = os.path.splitext(np_file)[0]
        caseids.append(filename[:-2])
        labels.append(filename[-1])

    print('>> Splitting caseids...')
    # split train and test
    caseids_train, caseids_test, _, _ = train_test_split(caseids, labels, test_size=test_size,
                                                                            stratify=labels, random_state=42)

    # save in files
    print('>> Saving splits...')
    np.save(os.path.join(path_to_save, 'train_caseids.npy'), caseids_train)
    np.save(os.path.join(path_to_save, 'test_caseids.npy'), caseids_test)
    print('>> Done')

    return


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        help='Lookup directory with data of all patients',
                        required=True,
                        type=str)
    parser.add_argument('--testsizeabsolute',
                        help='Size of test split as number of samples',
                        required=False,
                        type=int,
                        default=None)
    parser.add_argument('--testsizepercent',
                        help='Size of test split as percentage',
                        required=False,
                        type=float,
                        default=None)
    args = parser.parse_args()

    if args.testsizeabsolute > 0:
        test_size = args.testsizeabsolute
    elif args.testsizepercent > 0:
        test_size = args.testsizepercent
    else:
        raise argparse.ArgumentTypeError('Argument error: insert valid --testsizeabsolute or --testsizepercent')

    __split_data(args.dir, test_size)


if __name__ == '__main__':
    main()
