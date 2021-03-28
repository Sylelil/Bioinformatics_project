import argparse
import os
import sys
from os import path
from pathlib import Path
import tensorflow as tf
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import KFold
from tensorflow import keras
import numpy as np
import pandas as pd
from src.common import class_balancing, feature_concatenation, classification_methods, utils
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.pipeline import make_pipeline as sk_make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


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
    tile_features_train, tile_features_test, gene_features_train, gene_features_test = feature_concatenation.read_extracted_features()

    # TODO normalizzare geni e immagini prima di concatenarli?

    # concatenation of tile and gene features:
    if args.method == 'nn':
        gene_copy_ratio = 20 # TODO vedere se 20 va bene (tiles dim: 2048, genes dim: 100->100*20=2000)
    else:
        gene_copy_ratio = 1
    X_train, y_train = feature_concatenation.concatenate(tile_features_train, gene_features_train, gene_copy_ratio)
    X_test, y_test = feature_concatenation.concatenate(tile_features_test, gene_features_test, gene_copy_ratio)

    # prepare for training
    estimators = []

    # balance the training dataset
    if args.balancing:
        estimators.append(class_balancing.get_balancing_method(args.balancing, params))

    # get classifier and parameter grid
    classifier, param_grid = classification_methods.get_classifier_param_grid(args.method, params)

    # classification pipeline:
    estimators.append(classifier)

    if args.balancing:
        pipe = imb_make_pipeline(estimators)
    else:
        pipe = sk_make_pipeline(estimators)

    cv = KFold(n_splits=params['cv_grid_search_rank'])
    search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=params['scoring'],
                          refit=True)
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    y_pred = search.predict(X_test)

    print(classification_report_imbalanced(y_test, y_pred))


if __name__ == '__main__':
    main()