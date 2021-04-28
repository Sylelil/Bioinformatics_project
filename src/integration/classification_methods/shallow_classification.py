from pathlib import Path

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src.integration import class_balancing, utils, plots
from src.integration.classification_methods import common
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data_manipulation import concatenate_features

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def shallow_classifier(args, params, train_filepath, val_filepath, test_filepath):
    """
       Description: Train and test shallow classifier, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param train_filepath: train data path.
       :param val_filepath: validation data path.
       :param test_filepath: test data path.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = common.compute_scaling_pca(params, train_filepath, val_filepath, test_filepath)

    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = class_balancing.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)
    if args.balancing:
        metric = metrics.accuracy_score
    else:
        metric = metrics.matthews_corrcoef

    best_score = -1
    best_hyperparam = None
    if args.method == 'svc':
        print(">> Finding best hyperparameter C for LinearSVC...")
        grid = [0.0001, 0.001, 0.01, 0.1, 1] # C
    else:
        print(">> Finding best hyperparameter alpha for SGDClassifier...")
        grid = [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06] # alpha

    for hyperparam in grid:
        print(f"{'C' if args.method == 'svc' else 'alpha'}={hyperparam}:")
        if args.method == 'svc':
            classifier = LinearSVC(C=hyperparam,
                                   class_weight=('balanced' if args.balancing == 'weights' else None),
                                   random_state=params['general']['random_state'])
        else:
            classifier = SGDClassifier(alpha=hyperparam,
                                       max_iter=10, # np.ceil(10**6 / n_samples)
                                       class_weight=('balanced' if args.balancing == 'weights' else None),
                                       random_state=params['general']['random_state'])
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        score = metric(y_val, y_pred)
        print(f"    Validation {metric.__name__}: {score}")
        if score > best_score:
            best_score = score
            best_hyperparam = hyperparam

    print(f"Best {metric.__name__} ({'C' if args.method == 'svc' else 'alpha'}={best_hyperparam}): {best_score}")
    print(f">> Training with best {'LinearSVC' if args.method == 'svc' else 'SGDClassifier'} model...")
    if args.method == 'svc':
        best_classifier = LinearSVC(C=best_hyperparam,
                                    class_weight=('balanced' if args.balancing == 'weights' else None),
                                    random_state=params['general']['random_state'])
    else:
        best_classifier = SGDClassifier(alpha=best_hyperparam,
                                        max_iter=10,
                                        class_weight=('balanced' if args.balancing == 'weights' else None),
                                        random_state=params['general']['random_state'])
    best_classifier.fit(X_train, y_train)
    print(">> Testing...")
    y_pred_test = best_classifier.predict(X_test)
    y_pred_train = best_classifier.predict(X_train)

    print('Test scores:')
    test_scores = []
    for metric in common.METRICS_skl:
        test_scores.append((metric.__name__, metric(y_test, y_pred_test)))
    for name, value in test_scores:
        print(name, ': ', value)
    print()

    print(metrics.classification_report(y_test, y_pred_test))

    plots.plot_test_results(y_test, y_pred_test, y_train, y_pred_train)
    plt.show()

    # aggregate results for each patient in test dataset:
    common.aggregate_results_per_patient(y_pred_test)

    print('>> Done')
