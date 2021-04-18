import argparse
import os
from pathlib import Path

from common import split_data


def main():
    # Parse arguments from command line
    '''
    parser = argparse.ArgumentParser()
    test_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--dirimg',
                        help='Lookup directory with image data of all patients',
                        required=True,
                        type=str)
    parser.add_argument('--dirgene',
                        help='Lookup directory with gene data of all patients',
                        required=True,
                        type=str)
    parser.add_argument('--savedir',
                        help='Directory for saving split data',
                        required=True,
                        type=str)
    test_group.add_argument('--testsizeabsolute',
                            help='Size of test split as number of samples',
                            required=False,
                            type=int,
                            default=None)
    test_group.add_argument('--testsizepercent',
                            help='Size of test split as percentage',
                            required=False,
                            type=float,
                            default=None)
    args = parser.parse_args()

    # get int or float test size from arguments
    if args.testsizeabsolute is not None and args.testsizeabsolute > 0:
        test_size = args.testsizeabsolute
    elif args.testsizepercent is not None and args.testsizepercent > 0:
        test_size = args.testsizepercent
    else:
        raise argparse.ArgumentTypeError('Argument error: insert valid --testsizeabsolute or --testsizepercent')
    '''

    dirimg = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'images_not_split'
    dirgene = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'genes_not_split'
    savedir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results'
    test_size = float(0.2)
    val_size = float(0.2)
    val_size = val_size / (1 - test_size)

    if not os.path.exists(dirimg):
        print("%s not existing." % dirimg)
        exit()
    if not os.path.exists(dirgene):
        print("%s not existing." % dirgene)
        exit()
    if not os.path.exists(savedir):
        print("%s not existing." % savedir)
        exit()

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

    splitsdir = Path('assets') / 'data_splits'
    if not os.path.exists(Path('assets')):
        print("%s not existing." % Path('assets'))
        exit()
    if not os.path.exists(splitsdir):
        print("%s not existing." % splitsdir)
        exit()

    # split caseids in train, validation and test and save on file:
    caseids_train_val, caseids_test, labels_train_val, labels_test = split_data.split_caseids(lookup_dir=dirimg,
                                                                                              test_size=test_size,
                                                                                              save_dir=splitsdir,
                                                                                              nametrain='train_val',
                                                                                              nametest='test')
    caseids_train, caseids_val, _, _ = split_data.split_caseids(caseids_arg=caseids_train_val,
                                                                labels_arg=labels_train_val, test_size=val_size,
                                                                save_dir=splitsdir, nametrain='train', nametest='val')

    print('--------------------------')
    print(">> Splitting images files:")
    print('--------------------------')
    split_data.split_into_folders(dirimg, caseids_train, caseids_val, caseids_test, Path(savedir) / 'images')
    print('-------------------------')
    print(">> Splitting genes files:")
    print('-------------------------')
    split_data.split_into_folders(dirgene, caseids_train, caseids_val, caseids_test, Path(savedir) / 'genes')


if __name__ == '__main__':
    main()
