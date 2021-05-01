import configparser
import os
import sys
from collections import Counter

from imblearn.metrics import sensitivity_score, specificity_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from scipy import stats


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
            where n_samples is the number of samples and n_features is the number of selected features
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
        elif config['svm_t_rfe']['scoring'] == 'matt_coef':
            params['scoring'] = make_scorer(metrics.matthews_corrcoef)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'recall':
            params['scoring'] = make_scorer(metrics.recall_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'precision':
            params['scoring'] = make_scorer(metrics.precision_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'f1_score':
            params['scoring'] = make_scorer(metrics.f1_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'sensitivity':
            params['scoring'] = make_scorer(sensitivity_score)
            params['scoring_name'] = config['svm_t_rfe']['scoring']
        elif config['svm_t_rfe']['scoring'] == 'specificity':
            params['scoring'] = make_scorer(specificity_score)
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


def eval_asymmetry_and_kurt(df):
    n_skew_pos = 0
    n_skew_neg = 0
    n_kurt_1 = 0
    n_kurt_2 = 0

    for gene in tqdm(df.columns, desc="Evaluate asymmetry and kurt...", file=sys.stdout):
        if stats.skew(df[gene]) > 0.5:
            n_skew_pos += 1
        elif stats.skew(df[gene]) < -0.5:
            n_skew_neg += 1
        if stats.kurtosis(df[gene]) > 0:
            n_kurt_1 += 1
        elif stats.kurtosis(df[gene]) < 0:
            n_kurt_2 += 1

    print(">> Percentage of genes with asymmetric distribution (verso sx): %.3f" % (100 * (n_skew_pos / len(df.columns))))
    print(">> Percentage of genes with asymmetric distribution (verso dx): %.3f" % (100 * (n_skew_neg / len(df.columns))))
    print(">> Percentage of genes with platykurtic distribution: %.3f" % (100 * (n_kurt_2 / len(df.columns))))
    print(">> Percentage of genes with leptokurtic distribution: %.3f" % (100 * (n_kurt_1 / len(df.columns))))
    return n_skew_pos, n_skew_neg, n_kurt_1, n_kurt_2


def pca(X, y):

    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X)
    print('>> Cumulative explained variation for 2 principal components: {}'.format(
        np.sum(pca.explained_variance_ratio_)))

    df_pca = pd.DataFrame(
        dict(pca_one=pca_results[:, 0], pca_two=pca_results[:, 1], y=y))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="pca_one", y="pca_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_pca,
        legend="full",
        alpha=0.9,
    )
    plt.xlabel('pca_one')
    plt.ylabel('pca_two')
    plt.show()


def pca_3D(X, y):

    pca = PCA(n_components=3, random_state=42)
    pca_results = pca.fit_transform(X)
    print('>> Cumulative explained variation for 3 principal components: {}'.format(
        np.sum(pca.explained_variance_ratio_)))

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    scatter = ax.scatter(
        xs=pca_results[:, 0],
        ys=pca_results[:, 1],
        zs=pca_results[:, 2],
        c=y,
        cmap='tab10'
    )
    legend = ax.legend(*scatter.legend_elements())
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    ax.add_artist(legend)

    plt.show()


def tsne_pca(X, y):

    pca_50 = PCA(n_components=50, random_state=42)
    pca_result_50 = pca_50.fit_transform(X)
    print('>> Cumulative explained variation for 50 principal components: {}'.format(
        np.sum(pca_50.explained_variance_ratio_)))

    tsne = TSNE(n_components=2, random_state=42)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    df_tsne_pca = pd.DataFrame(
        dict(tsne_pca50_one=tsne_pca_results[:, 0], tsne_pca50_two=tsne_pca_results[:, 1], y=y))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="tsne_pca50_one", y="tsne_pca50_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_tsne_pca,
        legend="full",
        alpha=0.9,
    )
    plt.xlabel('tsne_pca50_one')
    plt.ylabel('tsne_pca50_two')
    plt.show()


def tsne(X, y):

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(
        dict(tsne_one=tsne_results[:, 0], tsne_two=tsne_results[:, 1], y=y))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="tsne_one", y="tsne_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_tsne,
        legend="full",
        alpha=0.9,
    )
    plt.xlabel('tsne_one')
    plt.ylabel('tsne_two')
    plt.show()


def tsne_3D(X, y):

    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(X)

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    scatter = ax.scatter(
        xs=tsne_results[:, 0],
        ys=tsne_results[:, 1],
        zs=tsne_results[:, 2],
        c=y,
        cmap='tab10'
    )
    legend = ax.legend(*scatter.legend_elements())
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    ax.add_artist(legend)

    plt.show()


def show_2D_svm_decision_boundary(params, X_train, y_train, X_test, y_test):

    C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = dict(svm__C=C_range)

    # Show decision boundary
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot
    scaler = StandardScaler()
    smt = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
    svm = SVC(kernel=params['kernel'])
    imba_pipeline = Pipeline([('scaler', scaler), ('smt', smt), ('svm', svm)])

    # define search
    cv = StratifiedKFold(n_splits=params['cv_grid_search_acc'], shuffle=True, random_state=params['random_state'])
    clf = GridSearchCV(estimator=imba_pipeline, param_grid=param_grid, scoring=params['scoring'], cv=cv, refit=True)
    clf.fit(X_train[:, :2], y_train)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    pred = clf.predict(X_test[:, :2])
    print(">> Test accuracy= %f" % accuracy_score(y_test, pred))

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()
