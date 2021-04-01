import argparse
import os
from pathlib import Path

from common import split_data


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    test_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--dir',
                        help='Lookup directory with data of all patients',
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

    data_splits_dir = Path('assets') / 'data_splits'

    if not os.path.exists(Path('assets')):
        os.mkdir(Path('assets'))

    if not os.path.exists(data_splits_dir):
        os.mkdir(data_splits_dir)

    # get int or float test size from arguments
    if args.testsizeabsolute is not None and args.testsizeabsolute > 0:
        test_size = args.testsizeabsolute
    elif args.testsizepercent is not None and args.testsizepercent > 0:
        test_size = args.testsizepercent
    else:
        raise argparse.ArgumentTypeError('Argument error: insert valid --testsizeabsolute or --testsizepercent')

    # split caseids and save on file:
    split_data.__split_caseids(args.dir, test_size, data_splits_dir)


if __name__ == '__main__':
    main()
