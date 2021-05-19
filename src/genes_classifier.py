import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import imblearn
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from config import paths
from genes import utils
from common.plots import plot_tsne_pca, plot_pca
from common.plots import plot_2D_svm_decision_boundary
from common.classification_report_utils import generate_classification_plots
from common.classification_report_utils import generate_classification_report
from common.classification_metrics import METRICS_skl
import numpy as np


def main():
    """
        Description: Main implementing different classification methods to classify
            patients in "tumor" or "normal", according to gene expression values
    """
    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)

    parser.add_argument('--classification_method',
                        help='Method to classify patients according to gene expression values',
                        choices=['svm', 'perceptron', 'sgd_classifier'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    if not os.path.exists(args.cfg) or (not os.path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    if not os.path.exists(paths.svm_t_rfe_selected_features_train) or \
            len(os.listdir(paths.svm_t_rfe_selected_features_train)) == 0:
        print("Directory " + str(paths.svm_t_rfe_selected_features_train) + " doesn't exists or is empty")
        exit(1)
    if not os.path.exists(paths.svm_t_rfe_selected_features_test) or \
            len(os.listdir(paths.svm_t_rfe_selected_features_test)) == 0:
        print("Directory " + str(paths.svm_t_rfe_selected_features_test) + " doesn't exists or is empty")
        exit(1)

    # path to save results:
    experiment_descr = f"CLF_{args.classification_method}"
    results_path = Path(paths.genes_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Read configuration file
    params = utils.read_config_file(args.cfg, args.classification_method)

    print("Reading gene expression data from disk..")
    X_train, y_train, t_train = utils.load_selected_genes(paths.svm_t_rfe_selected_features_train)
    X_val, y_val, t_val = utils.load_selected_genes(paths.svm_t_rfe_selected_features_val)
    X_test, y_test, t_test = utils.load_selected_genes(paths.svm_t_rfe_selected_features_test)

    print("Len X_train" + str(len(X_train)))
    print("Len X_val" + str(len(X_val)))
    print("Len X_test" + str(len(X_test)))
    print(Counter(y_val).keys())
    print(Counter(y_val).values())

    print(Counter(y_test).keys())
    print(Counter(y_test).values())

    # train + val
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = y_train + y_val

    # Data visualization
    print("\nData visualization:")
    scaler = StandardScaler()
    plot_tsne_pca(Path(paths.genes_classification_results_dir) / 'train_tsne_pca.png', scaler.fit_transform(X_train), y_train)
    plot_tsne_pca(Path(paths.genes_classification_results_dir) / 'test_tsne_pca.png', scaler.transform(X_test), y_test)

    classifier_str = ''
    param_grid = {}
    classifier = BaseEstimator()

    if args.classification_method == "svm":

        # Param grid
        C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = dict(svm__C=C_range)

        print("\nFit SVM with first 2 ranked features:")
        if params['smote']:
            scaler = StandardScaler()
            smt = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
            svm = SVC(kernel=params['kernel'])
            pipeline = imblearn.pipeline.Pipeline([('scaler', scaler), ('smt', smt), ('svm', svm)])
        else:
            scaler = StandardScaler()
            svm = SVC(kernel=params['kernel'])
            pipeline = sklearn.pipeline.Pipeline([('scaler', scaler), ('svm', svm)])

        # define search
        cv = StratifiedKFold(n_splits=params['cv_grid_search_acc'], shuffle=True, random_state=params['random_state'])
        clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=params['scoring'], cv=cv, refit=True)
        clf.fit(X_train[:, :2], y_train)

        pred = clf.predict(X_test[:, :2])
        print(">> Test accuracy= %f" % accuracy_score(y_test, pred))

        # Show decision boundary for svm trained on the first 2 features
        scaler = StandardScaler()
        plot_2D_svm_decision_boundary(Path(results_path) / '2D_svm_decision_boundary.png', clf, scaler.fit_transform(X_train[:, :2]), y_train, scaler.transform(X_test[:, :2]), y_test)

        # Fit svm model on all features
        # define model
        classifier = SVC(kernel=params['kernel'])
        classifier_str = 'svm'

    elif args.classification_method == "perceptron":

        # define grid
        max_iter = [10, 100, 1000, 10000]
        param_grid = dict(perceptron__max_iter=max_iter)

        # define model
        classifier = Perceptron(eta0=0.0001)
        classifier_str = 'perceptron'

    elif args.classification_method == "sgd_classifier":
        # define grid
        max_iter = [10, 100, 1000, 10000]
        param_grid = dict(SGDClassifier__max_iter=max_iter)

        # define model
        classifier = SGDClassifier(random_state=params['random_state'])
        classifier_str = 'SGDClassifier'

    else:
        sys.stderr.write("Invalid value for <classification_method>")
        exit(1)

    print("\nFit SVM with all features:")
    if params['smote']:
        scaler = StandardScaler()
        smt = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
        pipeline = imblearn.pipeline.Pipeline([('scaler', scaler), ('smt', smt), (classifier_str, classifier)])
    else:
        scaler = StandardScaler()
        pipeline = sklearn.pipeline.Pipeline([('scaler', scaler), (classifier_str, classifier)])

    # define search
    cv = StratifiedKFold(n_splits=params['cv_grid_search_acc'], shuffle=True, random_state=params['random_state'])
    clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=params['scoring'], cv=cv, refit=True)
    results = clf.fit(X_train, y_train)

    # best configuration
    print("\nResults with best parameters:")
    print(f">> Mean cross-validated score of the best_estimator ({params['scoring_name']}): {results.best_score_}")
    print('>> Config: %s' % results.best_params_)

    # summarize all
    print("\nAll configurations of parameters:")
    means = results.cv_results_['mean_test_score']
    clf_params = results.cv_results_['params']
    for mean, param in zip(means, clf_params):
        print(f">> Mean cross-validated score ({params['scoring_name']}) = {mean} with: {param}")

    print("Predict on train..")
    y_pred_train = clf.predict(X_train)
    print("Predict on test..")
    y_pred_test = clf.predict(X_test)

    test_scores = {}
    for metr in METRICS_skl:
        test_scores[metr.__name__] = metr(y_test, y_pred_test)

    # generate classification report:
    experiment_info = {}
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['Best hyperparameter'] = f"{'C' if args.classification_method == 'svm' else 'max_iter'}={results.best_params_}"
    experiment_info['Mean cross-validated score'] = f"{params['scoring_name']} = {results.best_score_}"
    generate_classification_report(results_path, y_test, y_pred_test, test_scores, experiment_info, patch_classification=False)

    # generate plots:
    generate_classification_plots(results_path, clf, X_test, y_test, X_train, y_train)
    print('>> Done')


if __name__ == "__main__":
    main()
