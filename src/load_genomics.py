import pandas as pd
import tensorflow as tf
import os
import sys


def main():

    path_ge = ""
    path_example_file = ""

    # Number of samples:
    n_samples_0 = sum(1 for x in os.listdir(path_ge) if x.endswith("_0.txt"))
    n_samples_1 = sum(1 for x in os.listdir(path_ge) if x.endswith("_1.txt"))

    # Number of features:
    with open(path_example_file) as f:
        n_features = sum(1 for _ in f)

    # Read samples:
    for path_ge in os.listdir(path_ge):
        with open(file_ge) as f:
            # todo

if __name__ == "__main__":
    main()

