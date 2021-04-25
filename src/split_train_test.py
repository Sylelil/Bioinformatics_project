import argparse
import os
from pathlib import Path

from common import split_data
from src.common import utils


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file path',
                        required=True,
                        type=str)
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
    # Parse arguments from command line and get params
    args = args_parse()
    params = utils.read_config_file(args.cfg)

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



    dirimg = params['paths']['images_dir']
    dirgene = params['paths']['genes_dir']
    savedir = params['paths']['split_data_dir']

    '''
    dirimg = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'images_not_split'
    dirgene = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'genes_not_split'
    savedir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results'
    test_size = float(0.2)
    val_size = float(0.2)
    val_size = val_size / (1 - test_size)
    '''

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

    caseid_splits_dir = Path('assets') / 'caseid_splits'
    if not os.path.exists(Path('assets')):
        print("%s not existing." % Path('assets'))
        exit()
    if not os.path.exists(caseid_splits_dir):
        print("%s not existing." % caseid_splits_dir)
        exit()

    # split caseids in train, validation and test and save on file:
    caseids_train_val, caseids_test, labels_train_val, labels_test = split_data.split_caseids(lookup_dir=dirgene,
                                                                                              test_size=test_size,
                                                                                              save_dir=caseid_splits_dir,
                                                                                              nametrain='train_val',
                                                                                              nametest='test')
    caseids_train, caseids_val, _, _ = split_data.split_caseids(caseids_arg=caseids_train_val,
                                                                labels_arg=labels_train_val,
                                                                test_size=val_size,
                                                                save_dir=caseid_splits_dir,
                                                                nametrain='train',
                                                                nametest='val')

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
