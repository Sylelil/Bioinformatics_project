import argparse
import sys
from os import path


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)

    parser.add_argument('--classification_method',
                        help='Method to classify patients according to features extracted from images',
                        choices=['svm', 'nn'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    # Read configuration file

    if args.classification_method == "svm":
        # TODO
        pass
    elif args.classification_method == "nn":
        # TODO
        pass
    else:
        sys.stderr.write("Invalid value for <classification_method>")
        exit(1)


if __name__ == "__main__":
    main()
