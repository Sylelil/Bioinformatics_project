import argparse
import os
import sys
from os import path
from pathlib import Path
import tensorflow as tf
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np
import pandas as pd
from src.common import class_balancing, feature_concatenation, classification_methods, utils
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.pipeline import make_pipeline as sk_make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)
    parser.add_argument('--method',
                        help='Classification method',
                        choices=['svm', 'nn', 'pca_nn'],
                        required=True,
                        type=str)
    parser.add_argument('--balancing',
                        help='Class balancing method',
                        choices=['random_upsampling', 'combined', 'smote', 'downsampling'],
                        required=False,
                        type=str)
    args = parser.parse_args()
    return args


def main():
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    params = utils.read_config_file(args.cfg, args.method)

    # Read features from file
    tile_features_train, tile_features_test, gene_features_train, gene_features_test = utils.read_extracted_features()

    # concatenation of tile and gene features:
    print('>> Concatenating gene and tile features...')
    if args.method == 'nn':
        gene_copy_ratio = 20 # TODO vedere se 20 va bene (tiles dim: 2048, genes dim: 100->100*20=2000)
    else:
        gene_copy_ratio = 1
    X_train, y_train, train_info = feature_concatenation.concatenate(tile_features_train, gene_features_train, gene_copy_ratio)
    X_test, y_test, test_info = feature_concatenation.concatenate(tile_features_test, gene_features_test, gene_copy_ratio)
    print('>> Done')

    # prepare for training
    print(f">> Starting training procedure with {args.method}...")
    estimators = []

    # add scaler
    estimators.append(StandardScaler())

    # add class balancing method
    if args.balancing:
        estimators.append(class_balancing.get_balancing_method(args.balancing, params))

    # add classifier and parameter grid
    classifier, param_grid = classification_methods.get_classifier_param_grid(args.method, params)
    estimators.append(classifier)

    # create pipeline
    if args.balancing:
        pipe = imb_make_pipeline(estimators)
    else:
        pipe = sk_make_pipeline(estimators)

    test_outer_results = []
    train_outer_results = []   # TODO CI SERVE??
    best_hyperparams = []

    print(">> Performing nested cross validation for unbiased error estimation...")
    cv_outer = KFold(n_splits=params['cv_outer_n_splits'], shuffle=True, random_state=params['random_state'])
    # nested cross validation for unbiased error estimation:
    for train_ix, test_ix in tqdm(cv_outer.split(X_train, y_train)):
        # split data in k_outer folds (one is test, the rest is trainval) for outer loop
        X_train_cv, X_test_cv = X_train[train_ix, :], X_train[test_ix, :]
        y_train_cv, y_test_cv = y_train[train_ix], y_train[test_ix]
        # inner cross validation procedure for grid search of best hyperparameters:
        # trainval will be split in k_inner folds (one is val, the rest is train)
        # use train and val to find best model
        cv_inner = KFold(n_splits=params['cv_inner_n_splits'], shuffle=True, random_state=params['random_state'])
        search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv_inner, n_jobs=-1, scoring=params['scoring'],
                              refit=True)
        # outer cross validation procedure to evaluate the performance of the best estimator:
        # fit the best model on the whole trainval
        search.fit(X_train_cv, y_train_cv)
        best_model = search.best_estimator_
        # evaluate the performance of the model on test
        y_test_pred = best_model.predict(X_test_cv)
        test_score = params['scoring'](y_test_cv, y_test_pred)
        y_train_pred = search.predict(X_train_cv)
        train_score = params['scoring'](y_train_cv, y_train_pred)
        test_outer_results.append(test_score)
        train_outer_results.append(train_score)
        best_hyperparams.append(search.best_params_)

    # calculate the mean score over all K outer folds, and report as the generalization error
    global_test_score = np.mean(test_outer_results)
    global_test_std = np.std(test_outer_results)
    global_train_score = np.mean(train_outer_results)
    global_train_std = np.std(train_outer_results)
    print(f"\nTest score {params['scoring'].__name__} = {str(global_test_score)} ({str(global_test_std)})")
    print(f"Train score {params['scoring'].__name__} = {str(global_train_score)} ({str(global_train_std)})")
    print("list of best hyperparameters to check stability: ")
    print(best_hyperparams)

    # simple cross validation to find the best model:
    print(">> Performing simple cross validation to find the best model...")
    cv = KFold(n_splits=params['cv_simple_n_splits'], shuffle=True, random_state=params['random_state'])
    search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=params['scoring'], refit=True)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # predict on test
    print(">> Predicting on test...")
    y_pred = best_model.predict(X_test)
    final_test_score = params['scoring'](y_test, y_pred)
    print(f"Final test score {params['scoring'].__name__} = {final_test_score}")
    print('Imbalanced classification report:')
    print(classification_report_imbalanced(y_test, y_pred))

    print('>> Done')


if __name__ == '__main__':
    main()