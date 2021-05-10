from pathlib import Path

import numpy as np


def csv_data_generator(inputPath, batchsize, scaler, n_features_images, balancer=None, dataset_name="train", mode='train'):
    """
       Description: Data generator from csv file.
       :param n_features_images: number of features of images to be considered.
       :param inputPath: path where input data is located.
       :param batchsize: size of each batch.
       :param scaler: standard scaler model.
       :param dataset_name: train or test mode.
       :param balancer: class balancing model.
       :param mode: mode (default: 'train')
       :yields: batch data and labels
    """
    filepath_data = Path(inputPath) / f'x_{dataset_name}.csv'
    filepath_labels = Path(inputPath) / f'y_{dataset_name}.csv'

    # open the CSV file for reading
    with open(filepath_data, "r") as f_data, open(filepath_labels, "r") as f_labels:
        while True:
            # initialize our batches of data and labels
            X = []
            y = []
            # keep looping until we reach our batch size

            while len(X) < batchsize:
                # attempt to read the next line of the CSV file
                line = f_data.readline()
                label = f_labels.readline()
                # check to see if the line is empty, indicating we have
                # reached the end of the file
                if line == "":
                    # reset the file pointer to the beginning of the file
                    # and re-read the line
                    f_data.seek(0)
                    f_labels.seek(0)
                    line = f_data.readline()
                    label = f_labels.readline()
                    # if we are evaluating we should now break from our
                    # loop to ensure we don't continue to fill up the
                    # batch from samples at the beginning of the file
                    if mode == "eval":
                        break
                # extract label and data
                line = line.strip().split(",")
                data = np.array([np.float64(x) for x in line])
                if n_features_images:
                    data = data[:n_features_images]
                # update our corresponding batches lists
                X.append(data)
                y.append(label)

            # apply scaler to data
            X = scaler.transform(X)

            # if the class balancing object is not None, apply it
            if balancer is not None:
                X, y = balancer.fit_resample(X, y)

            X, y = np.array(X), np.array(y)

            # yield the batch to the calling function
            yield X, y