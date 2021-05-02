import os
from pathlib import Path

from pandas import DataFrame
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from tqdm import tqdm

from config import paths
from src.integration import utils, plots
from src.integration.classification_methods import common
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data_manipulation import concatenate_features


def get_classifier(hyperparam, method_name, balancing, random_state):
    if method_name == 'linearsvc':
        classifier = LinearSVC(C=hyperparam,
                               class_weight=('balanced' if balancing == 'weights' else None),
                               random_state=random_state)
    else:
        classifier = SGDClassifier(alpha=hyperparam,
                                   max_iter=20,  # np.ceil(10**6 / n_samples)
                                   class_weight=('balanced' if balancing == 'weights' else None),
                                   random_state=random_state,
                                   average=True) # Averaged SGD works best with a larger number of features and a higher eta0
    return classifier


def shallow_classifier(args, params, train_filepath, val_filepath, test_filepath):
    """
       Description: Train and test shallow classifier, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param train_filepath: train data path.
       :param val_filepath: validation data path.
       :param test_filepath: test data path.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = utils.compute_scaling_pca(params, train_filepath, val_filepath, test_filepath)

    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = common.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)
    if args.balancing:
        metric = metrics.accuracy_score
    else:
        metric = metrics.matthews_corrcoef

    if args.classification_method == 'linearsvc':
        print(">> Finding best hyperparameter C for LinearSVC...")
        grid = [0.0001, 0.001, 0.01, 0.1, 1] # C
    else:
        print(">> Finding best hyperparameter alpha for SGDClassifier...")
        grid = [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06] # alpha

    best_score = -1
    best_hyperparam = None
    for hyperparam in grid:
        print(f"{'C' if args.classification_method == 'linearsvc' else 'alpha'}={hyperparam}:")
        classifier = get_classifier(hyperparam=hyperparam,
                                    method_name=args.classification_method,
                                    balancing=args.balancing,
                                    random_state=params['general']['random_state'])
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        score = metric(y_val, y_pred)
        print(f"    Validation {metric.__name__}: {score}")
        if score > best_score:
            best_score = score
            best_hyperparam = hyperparam

    # evaluate the performance of the model on test
    print()
    print(
        f"Best {metric.__name__} ({'C' if args.classification_method == 'linearsvc' else 'alpha'}={best_hyperparam}): {best_score}")
    print()
    print(f">> Training with best {'LinearSVC' if args.classification_method == 'linearsvc' else 'SGDClassifier'} model...")
    best_classifier = get_classifier(hyperparam=best_hyperparam,
                                     method_name=args.classification_method,
                                     balancing=args.balancing,
                                     random_state=params['general']['random_state'])

    best_classifier.fit(X_train, y_train)
    print(">> Testing...")
    print()
    y_pred_test = best_classifier.predict(X_test)
    y_pred_train = best_classifier.predict(X_train)

    test_scores = {}
    for metric in common.METRICS_skl:
        test_scores[metric.__name__] = metric(y_test, y_pred_test)

    # path to save results:
    experiment_descr = f"CLF_{args.classification_method}_PCA_{params['pca']['n_components']}_BAL_{args.balancing}"
    results_path = Path(paths.integration_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # generate classification report:
    experiment_info = {}
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['PCA n. components'] = str(params['pca']['n_components'])
    experiment_info['Class balancing method'] = str(args.balancing)
    experiment_info['Best hyperparameter'] = f"{'C' if args.classification_method == 'linearsvc' else 'alpha'}={best_hyperparam}"
    experiment_info['Best validation score'] = f"{metric.__name__}={best_score}"
    utils.generate_classification_report(results_path, y_test, y_pred_test, test_scores, experiment_info)

    # generate plots:
    #utils.generate_classification_plots(results_path, best_classifier, X_train, y_train, X_test, y_test)
    utils.generate_classification_plots(results_path, y_test, y_pred_test, y_train, y_pred_train)
    print('>> Done')
