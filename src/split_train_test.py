import argparse
import os
from pathlib import Path
from src.data_manipulation import split_data
from config import paths


def args_parse():
    """
       Description: Parse command-line arguments.
       :returns: arguments parser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--testsizepercent',
                        help='Size of test split as percentage value between 0 and 1',
                        required=True,
                        type=float)
    parser.add_argument('--valsizepercent',
                        help='Size of validation split as percentage value between 0 and 1',
                        required=True,
                        type=float)
    args = parser.parse_args()
    return args


def main():
    """
       Description: Split data into train, validation and test splits and copy files in respective folders.
    """
    # Parse arguments from command line and get params
    args = args_parse()

    # get val and test size from arguments
    if 0 < args.testsizepercent < 1:
        test_size = args.testsizepercent
    else:
        raise argparse.ArgumentTypeError('Argument error: invalid --testsizepercent')
    if 0 < args.valsizepercent < 1:
        val_size_absolute = args.valsizepercent  # absolute percentage
        val_size = val_size_absolute / (1 - test_size)  # relative percentage to train+val split
    else:
        raise argparse.ArgumentTypeError('Argument error: invalid --valsizepercent')

    dirimg = paths.images_dir
    dirgene = paths.genes_dir
    savedir = paths.split_data_dir

    if not os.path.exists(dirimg):
        print("%s not existing." % dirimg)
        exit()
    if not os.path.exists(dirgene):
        print("%s not existing." % dirgene)
        exit()
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if not os.path.exists(Path(savedir) / 'genes'):
        os.mkdir(Path(savedir) / 'genes')
    if not os.path.exists(Path(savedir) / 'genes' / 'train'):
        os.mkdir(Path(savedir) / 'genes' / 'train')
    if not os.path.exists(Path(savedir) / 'genes' / 'test'):
        os.mkdir(Path(savedir) / 'genes' / 'test')
    if not os.path.exists(Path(savedir) / 'genes' / 'val'):
        os.mkdir(Path(savedir) / 'genes' / 'val')

    if not os.path.exists(Path(savedir) / 'images'):
        os.mkdir(Path(savedir) / 'images')
    if not os.path.exists(Path(savedir) / 'images' / 'train'):
        os.mkdir(Path(savedir) / 'images' / 'train')
    if not os.path.exists(Path(savedir) / 'images' / 'test'):
        os.mkdir(Path(savedir) / 'images' / 'test')
    if not os.path.exists(Path(savedir) / 'images' / 'val'):
        os.mkdir(Path(savedir) / 'images' / 'val')

    filename_splits_dir = paths.filename_splits_dir

    if not os.path.exists(filename_splits_dir):
        os.makedirs(filename_splits_dir)

    # split caseids in train, validation and test and save on file:
    filenames_train_val, filenames_test, labels_train_val, labels_test = split_data.split_filenames(lookup_dir=dirgene,
                                                                                                    test_size=test_size,
                                                                                                    save_dir=filename_splits_dir,
                                                                                                    nametrain='train_val',
                                                                                                    nametest='test')
    filenames_train, filenames_val, _, _ = split_data.split_filenames(filenames_arg=filenames_train_val,
                                                                      labels_arg=labels_train_val,
                                                                      test_size=val_size,
                                                                      save_dir=filename_splits_dir,
                                                                      nametrain='train',
                                                                      nametest='val')

    print('--------------------------')
    print(">> Splitting images files:")
    print('--------------------------')
    split_data.split_into_folders(dirimg, filenames_train, filenames_val, filenames_test, Path(savedir) / 'images')
    print('-------------------------')
    print(">> Splitting genes files:")
    print('-------------------------')
    split_data.split_into_folders(dirgene, filenames_train, filenames_val, filenames_test, Path(savedir) / 'genes')


if __name__ == '__main__':
    main()
