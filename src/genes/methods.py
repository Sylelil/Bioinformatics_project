import configparser
import os
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import make_scorer
from tqdm import tqdm


def read_genes_from_folder(lookup_dir):
    """
        Description: Reading gene expression data from specified directory.
            The directory contains one .txt file for each case_id (patient).
            Each .txt file contains the gene expression values (features) for one specific case_id.

        :param lookup_dir: Path
            directory containing gene expression data
        :return df_patients: DataFrame, shape = [n_samples, n_features],
             where n_samples is the number of samples and n_features is the number of features.
             - DataFrame.columns: contains the gene names
             - DataFrame.index: contains the case_ids
        :return y: array-like, shape = [n_samples]
            labels (0/1)
    """
    X = pd.DataFrame()
    y = []
    for file_name in tqdm(os.listdir(lookup_dir), desc=">> Reading genes data...", file=sys.stdout):
        file_path = os.path.join(lookup_dir, file_name)
        with open(file_path) as f:
            patient_df = pd.read_csv(f, sep="\t", header=None, index_col=0, names=[file_name.replace(".txt", "")])
            patient_df = pd.DataFrame.transpose(patient_df)
            X = X.append(patient_df)
            y.append(0 if file_name.endswith("_0.txt") else 1)

    return X, y


def load_selected_genes(selected_features_dir):
    """
        Description: Reading gene expression values (features) of genes selected by feature selection method.
            The directory contains one .npy file for each case_id (patient).
            Each .npy file contains the gene expression values for one specific case_id.

        :param selected_features_dir: Path
            directory containing the gene expression data
        :returns X: 2D-numpy array, shape = [n_samples, n_selected_features],
            where n_samples is the number of samples and n_selected_features is the number of selected features
        :return y: array-like, shape = [n_samples]
            labels (0/1)
        :return targets: array-like, shape = [n_samples]
            case_ids
    """
    X = []
    y = []
    case_ids = []
    for patient_file in tqdm(os.listdir(selected_features_dir), desc=">> Reading selected genes...", file=sys.stdout):
        patient_features = np.load(os.path.join(selected_features_dir, patient_file))
        case_id = os.path.splitext(patient_file)[0]
        target = case_id[-1:]
        case_ids.append(case_id)
        X.append(patient_features)
        y.append(int(target))

    X = np.asarray(X)
    return X, y, case_ids


def save_selected_genes(X, extracted_features_dir):
    """
        Description: Saving to disk gene expression values of genes selected by feature selection algorithm.
            The directory contains one .npy file for each case_id (patient).
            Each .npy file contains the gene expression values for one specific case_id.

        :param X: DataFrame, shape = [n_samples, n_selected_features],
            where n_samples is the number of samples and n_features is the number of selected features.
        :param extracted_features_dir: Path
            directory to save gene expression data
    """

    for index, row in X.iterrows():
        row = np.asarray(row)
        np.save(os.path.join(extracted_features_dir, index + '.npy'), row)

    print(">> Features saved to " + str(extracted_features_dir))


def read_config_file(config_file_path, section):
    """
        Description: Reading configuration file for genes
        :param config_file_path: Path
            configuration file
        :param section: String
            configuration file section
        :return params: Dictionary
            configuration file parameters
       """
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file_path)

    if section == 'welch_t':
        params['alpha'] = config.getfloat('welch_t_test', 'alpha')

    elif section == 'svm_t_rfe':
        params['random_state'] = config.getint('general', 'random_state')
        params['sampling_strategy'] = config.getfloat('general', 'sampling_strategy')
        params['alpha'] = config.getfloat('svm_t_rfe', 'alpha')
        params['theta'] = config.getfloat('svm_t_rfe', 'theta')
        params['cv_grid_search_rank'] = config.getint('svm_t_rfe', 'cv_grid_search_rank')
        params['cv_grid_search_acc'] = config.getint('svm_t_rfe', 'cv_grid_search_acc')
        params['cv_outer'] = config.getint('svm_t_rfe', 'cv_outer')
        params['top_ranked'] = config.getint('svm_t_rfe', 'top_ranked')
        params['t_stat_threshold'] = config.getfloat('svm_t_rfe', 't_stat_threshold')
        if config['svm_t_rfe']['scoring'] == 'accuracy':
            params['scoring'] = make_scorer(metrics.accuracy_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'recall':
            params['scoring'] = make_scorer(metrics.recall_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        else:
            sys.stderr.write("Invalid value for <scoring> in config file")
            exit(1)
        if config['svm_t_rfe']['kernel'] == 'linear' or config['svm_t_rfe']['kernel'] == 'rbf':
            params['kernel'] = config['svm_t_rfe']['kernel']
        else:
            sys.stderr.write("Invalid value for <kernel> in config file")
            exit(1)
    elif section == 'svm':
        params['random_state'] = config.getint('general', 'random_state')
        params['sampling_strategy'] = config.getfloat('general', 'sampling_strategy')
        params['cv_grid_search_acc'] = config.getint('svm', 'cv_grid_search_acc')
        if config['svm']['kernel'] == 'linear' or config['svm']['kernel'] == 'rbf':
            params['kernel'] = config['svm']['kernel']
        else:
            sys.stderr.write("Invalid value for <kernel> in config file")
            exit(1)
        if config['svm']['scoring'] == 'accuracy':
            params['scoring'] = make_scorer(metrics.accuracy_score)
        else:
            sys.stderr.write("Invalid value for <scoring> in config file")
            exit(1)
    elif section == 'perceptron':
        params['random_state'] = config.getint('general', 'random_state')
        params['sampling_strategy'] = config.getfloat('general', 'sampling_strategy')
        params['cv_grid_search_acc'] = config.getint('perceptron', 'cv_grid_search_acc')
        if config['perceptron']['scoring'] == 'accuracy':
            params['scoring'] = make_scorer(metrics.accuracy_score)
        elif config['perceptron']['scoring'] == 'recall':
            params['scoring'] = make_scorer(metrics.recall_score)
        else:
            sys.stderr.write("Invalid value for <scoring> in config file")
            exit(1)
    elif section == 'sgd_classifier':
        params['random_state'] = config.getint('general', 'random_state')
        params['sampling_strategy'] = config.getfloat('general', 'sampling_strategy')
        params['cv_grid_search_acc'] = config.getint('sgd_classifier', 'cv_grid_search_acc')
        if config['sgd_classifier']['scoring'] == 'accuracy':
            params['scoring'] = make_scorer(metrics.accuracy_score)
        elif config['sgd_classifier']['scoring'] == 'recall':
            params['scoring'] = make_scorer(metrics.recall_score)
        else:
            sys.stderr.write("Invalid value for <scoring> in config file")
            exit(1)
    else:
        sys.stderr.write("Invalid value for <section> in config file")
        exit(1)

    return params
