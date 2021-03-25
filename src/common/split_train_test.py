import numpy as np
from sklearn.model_selection import train_test_split

def split_data(lookup_dir, test_size):
    """
       Split data into random train and test subsets according to specified test size with stratification. 
       Saves splitted caseids into two files in 'assets\train_test_split' folder.
       Args:
            lookup_dir: lookup directory with data to be split.
            param test_size: float or int size of test subset.
    """
    # read case ids and labels of all samples
    caseids = []
    labels = []
    for np_file in os.listdir(lookup_dir):
        filename = os.path.splitext(np_file)[0]
        caseids.append(filename[:-2])
        labels.append(filename[-1])
    caseids = np.array(caseids)
    labels = np.array(labels)

    # split
    caseids_train, caseids_test, _, _ = train_test_split(caseids, labels, test_size=test_size, stratify=labels, random_state=42)

    # save in files
    path_to_save = Path('assets') / 'train_test_split'
    np.save(os.path.join(path_to_save, 'train_caseids.npy'), caseids_train)
    np.save(os.path.join(path_to_save, 'test_caseids.npy'), caseids_test)

    return 


def get_split_caseids():
    """
        Get train and test splits of caseids saved in 'assets\train_test_split' folder.
        Returns:
            train_caseids: list of caseids of train split.
            test_caseids: list of caseids of test split.
    """
    lookup_dir = Path('assets') / 'train_test_split'
    file_path_train = os.path.join(lookup_dir, 'train_caseids.npy')
    file_path_test = os.path.join(lookup_dir, 'test_caseids.npy')
    train_caseids = np.load(file_path_train)
    test_caseids = np.load(file_path_test)
    return train_caseids,test_caseids

def get_split_data(lookup_dir):
    """
       Split data according to splits saved in 'assets\train_test_split' folder.
       Args:
            lookup_dir: lookup directory with data to be split.
        Returns:
            X_train, X_test, y_train, y_test: train and test subsets of data and labels
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # get splitted caseids
    train_caseids,test_caseids = get_split_caseids_labels()

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