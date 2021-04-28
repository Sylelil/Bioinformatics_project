import numpy as np


def csv_data_generator(inputPath, batchsize, scaler, mode="train", balancer=None):
    """
       Description: Return method corresponding to the parameter 'method'.
       :param inputPath: path where input data is located.
       :param batchsize: size of each batch.
       :param scaler: standard scaler model.
       :param mode: train or test mode.
       :param balancer: class balancing model.
       :yields: batch data and labels
    """
    # open the CSV file for reading
    with open(inputPath, "r") as f:
        f.readline()  # skip first line of file with header
        while True:
            # initialize our batches of data and labels
            X = []
            y = []
            # keep looping until we reach our batch size

            while len(X) < batchsize:
                # attempt to read the next line of the CSV file
                line = f.readline()
                # check to see if the line is empty, indicating we have
                # reached the end of the file
                if line == "":
                    # reset the file pointer to the beginning of the file
                    # and re-read the line
                    f.seek(0)
                    f.readline()  # skip first line of file with header
                    line = f.readline()
                    # if we are evaluating we should now break from our
                    # loop to ensure we don't continue to fill up the
                    # batch from samples at the beginning of the file
                    if mode == "eval":
                        break
                # extract label and data
                line = line.strip().split(",")
                label = int(line[-1])
                data = np.array([np.float64(x) for x in line[:-1]])
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
