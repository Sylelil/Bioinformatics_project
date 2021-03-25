import argparse
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_data(lookup_dir, test_size, val_size):
    """
       Split data into random train, test and validation subsets according to specified sizes with stratification.
       Saves splitted caseids into three files in 'assets\train_test_split' folder.
       Args:
            lookup_dir: lookup directory with data to be split.
            test_size: float or int size of test subset.
            val_size: float or int size of validation subset.
    """

    print('>> Reading data...')
    # read case ids and labels of all samples
    caseids = []
    labels = []
    for np_file in os.listdir(lookup_dir):
        filename = os.path.splitext(np_file)[0]
        caseids.append(filename[:-2])
        labels.append(filename[-1])

    print('>> Splitting caseids...')
    # split train+val and test
    caseids_train_val, caseids_test, labels_train_val, _ = train_test_split(caseids, labels, test_size=test_size,
                                                                            stratify=labels, random_state=42)

    # split train and val
    if isinstance(val_size, int):
        caseids_train, caseids_val, _, _ = train_test_split(caseids_train_val, labels_train_val, test_size=val_size,
                                                            stratify=labels_train_val,
                                                            random_state=42)
    else:
        val_size_percent = (val_size / (100 - test_size)) * 100
        caseids_train, caseids_val, _, _ = train_test_split(caseids_train_val, labels_train_val, test_size=val_size_percent,
                                                            stratify=labels_train_val,
                                                            random_state=42)

    print('>> Saving splits...')
    # save in files
    path_to_save = Path('..') / '..' / 'assets' / 'data_splits'
    np.save(os.path.join(path_to_save, 'train_caseids.npy'), caseids_train)
    np.save(os.path.join(path_to_save, 'val_caseids.npy'), caseids_val)
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
                        default=-1)
    parser.add_argument('--testsizepercent',
                        help='Size of test split as percentage',
                        required=False,
                        type=float,
                        default=-1)
    parser.add_argument('--valsizeabsolute',
                        help='Size of val split as number of samples',
                        required=False,
                        type=int,
                        default=-1)
    parser.add_argument('--valsizepercent',
                        help='Size of val split as percentage',
                        required=False,
                        type=float,
                        default=-1)
    args = parser.parse_args()

    if args.testsizeabsolute > 0:
        test_size = args.testsizeabsolute
    elif args.testsizepercent > 0:
        test_size = args.testsizepercent
    else:
        raise argparse.ArgumentTypeError('Missing argument: insert valid --testsizeabsolute or --testsizepercent')

    if args.valsizeabsolute > 0:
        val_size = args.valsizeabsolute
    elif args.valsizepercent > 0:
        val_size = args.valsizepercent
    else:
        raise argparse.ArgumentTypeError('Missing argument: insert valid --valsizeabsolute or --valsizepercent')

    split_data(args.dir, test_size, val_size)

if __name__ == '__main__':
    main()