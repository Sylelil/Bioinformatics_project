import argparse
import os
import sys
from collections import Counter
from os import path
from imblearn.metrics import classification_report_imbalanced, sensitivity_score, specificity_score
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, \
    plot_confusion_matrix, plot_roc_curve, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from config import paths
from genes import methods
import numpy as np


def main():
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

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    if not os.path.exists(paths.svm_t_rfe_selected_features_train) or \
            len(os.listdir(paths.svm_t_rfe_selected_features_train)) == 0:
        print("Directory " + str(paths.svm_t_rfe_selected_features_train) + " doesn't exists or is empty")
        exit(1)
    if not os.listdir(paths.svm_t_rfe_selected_features_test) or \
            len(os.listdir(paths.svm_t_rfe_selected_features_test)) == 0:
        print("Directory " + str(paths.svm_t_rfe_selected_features_test) + " doesn't exists or is empty")
        exit(1)

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.classification_method)

    print("Reading gene expression data from disk..")
    X_train, y_train, t_train = methods.load_selected_genes(paths.svm_t_rfe_selected_features_train)
    X_val, y_val, t_val = methods.load_selected_genes(paths.svm_t_rfe_selected_features_val)
    X_test, y_test, t_test = methods.load_selected_genes(paths.svm_t_rfe_selected_features_test)

    # train + val
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = y_train + y_val

    # Data visualization
    print("\nData visualization:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    methods.tsne_pca(X_train_scaled, y_train)
    methods.tsne_pca(X_test_scaled, y_test)

    classifier_str = ''
    param_grid = {}
    classifier = BaseEstimator()

    if args.classification_method == "svm":

        # Show decision boundary for svm trained on the first 2 features
        print("\nFit SVM with first 2 ranked features:")
        methods.show_2D_svm_decision_boundary(params, X_train, y_train, X_test, y_test)

        # Fit svm model on all features

        # define grid
        C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = dict(svm__C=C_range)

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
    scaler = StandardScaler()
    smt = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
    imba_pipeline = Pipeline([('scaler', scaler), ('smt', smt), (classifier_str, classifier)])

    # define search
    cv = StratifiedKFold(n_splits=params['cv_grid_search_acc'], shuffle=True, random_state=params['random_state'])
    clf = GridSearchCV(estimator=imba_pipeline, param_grid=param_grid, scoring=params['scoring'], cv=cv, refit=True)
    results = clf.fit(X_train, y_train)

    # best configuration
    print("\nResults with best parameters:")
    print('>> Mean Cross-Validation Accuracy: %.3f' % results.best_score_)
    print('>> Config: %s' % results.best_params_)

    # summarize all
    print("\nAll configurations of parameters:")
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">> Mean Cross-Validation Accuracy = %.3f with: %r" % (mean, param))
    pred = clf.predict(X_test)

    print("\nMetrics on test set:")
    print(">> Test accuracy: %f" % accuracy_score(y_test, pred))
    average_precision = average_precision_score(y_test, pred, pos_label=1)
    precision = precision_score(y_test, pred, average='binary', pos_label=1)
    recall = recall_score(y_test, pred, average='binary', pos_label=1)
    sensitivity = sensitivity_score(y_test, pred, average='binary', pos_label=1)
    specificity = specificity_score(y_test, pred, average='binary', pos_label=1)

    print('>> Average precision-recall score: %f' % average_precision)
    print('>> Precision score: %f' % precision)
    print('>> Recall score: %f' % recall)
    print('>> sensitivity: %f' % sensitivity)
    print('>> specificity: %f' % specificity)

    print(classification_report_imbalanced(y_test, pred))
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()
    plot_roc_curve(clf, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main()
