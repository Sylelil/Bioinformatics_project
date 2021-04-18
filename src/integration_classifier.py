import argparse
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import LinearSVC
from src.common import class_balancing, classification_methods, utils
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.keras import BalancedBatchGenerator
from sklearn.pipeline import make_pipeline as sk_make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sklearn.decomposition import IncrementalPCA


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file path',
                        required=True,
                        type=str)
    parser.add_argument('--method',
                        help='Classification method',
                        choices=['svc', 'sgd', 'nn', 'pca_nn'],
                        required=True,
                        type=str)
    parser.add_argument('--balancing',
                        help='Class balancing method',
                        choices=['random_upsampling', 'combined', 'smote', 'downsampling', 'weights'],
                        required=False,
                        type=str)
    args = parser.parse_args()
    return args


def svc_classification(args, params, train_filepath, val_filepath, test_filepath):
    batchsize = params['batchsize']

    print(">> Fitting scaler...")
    scaler = StandardScaler()
    for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_train_chunk = chunk.iloc[:, :-1]
        scaler.partial_fit(X_train_chunk)

    ipca = IncrementalPCA(n_components=params['n_components'])
    X_train_ipca = []
    y_train_ipca = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    print(">> Transforming train data with scaler and fitting incremental pca...")
    for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_train_chunk = chunk.iloc[:, :-1]
        y_train_chunk = chunk['label']
        X_train_chunk_scaled = scaler.transform(X_train_chunk)
        ipca.partial_fit(X_train_chunk_scaled)
    print(">> Transforming train data with incremental pca...")
    for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_train_chunk = chunk.iloc[:, :-1]
        y_train_chunk = chunk['label']
        X_train_chunk_ipca = ipca.transform(X_train_chunk)
        X_train_ipca.extend(X_train_chunk_ipca)
        y_train_ipca.extend(y_train_chunk)
    print(">> Transforming validation data with incremental pca...")
    for chunk in tqdm(pd.read_csv(val_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_val_chunk = chunk.iloc[:, :-1]
        y_val_chunk = chunk['label']
        X_val_chunk_scaled = scaler.transform(X_val_chunk)
        X_val_chunk_ipca = ipca.transform(X_val_chunk_scaled)
        X_val.extend(X_val_chunk_ipca)
        y_val.extend(y_val_chunk)
    print(">> Transforming test data with incremental pca...")
    for chunk in tqdm(pd.read_csv(test_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
        X_test_chunk = chunk.iloc[:, :-1]
        y_test_chunk = chunk['label']
        X_test_chunk_scaled = scaler.transform(X_test_chunk)
        X_test_chunk_ipca = ipca.transform(X_test_chunk_scaled)
        X_test.extend(X_test_chunk_ipca)
        y_test.extend(y_test_chunk)

    if args.balancing and args.balancing != 'weights':
            print(">> Applying class balancing method...")
            balancer = class_balancing.get_balancing_method(args.balancing, params)
            X_train, y_train = balancer.fit_resample(X_train_ipca, y_train_ipca)
    else:
        X_train, y_train = X_train_ipca, y_train_ipca

    print(">> Finding best hyperparameter C for linear svc...")
    C_grid = [0.0001, 0.001, 0.01, 0.1, 1]
    best_score = -1
    best_C = None
    for C in C_grid:
        print(f'C={C}:')
        if args.balancing == 'weights':
            linearsvc = LinearSVC(random_state=params['random_state'], C=C, class_weight='balanced')
        else:
            linearsvc = LinearSVC(random_state=params['random_state'], C=C)
        linearsvc.fit(X_train, y_train)
        y_pred = linearsvc.predict(X_val)
        score = params['scoring'](y_val, y_pred)
        print(f"    Score {params['scoring'].__name__}: {score}")
        if score > best_score:
            best_score = score
            best_C = C

    print(f"Best score {params['scoring'].__name__} (C={best_C}): {best_score}")
    print(">> Training with best linear svc model...")
    best_linearsvc = LinearSVC(random_state=params['random_state'], C=best_C, class_weight='balanced')
    best_linearsvc.fit(X_train, y_train)
    print(">> Testing...")
    y_pred_test = best_linearsvc.predict(X_test)
    test_score = params['scoring'](y_test, y_pred_test)
    print(f"Test score: {params['scoring'].__name__} = {test_score}")

    print(classification_report(y_test, y_pred_test))
    print('>> Done')


def nn_classification(args, params, train_filepath, val_filepath, test_filepath):
    pass


def main():
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    params = utils.read_config_file(args.cfg)

    train_filepath = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'train' / 'concat_data.csv'
    val_filepath = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'val' / 'concat_data.csv'
    test_filepath = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'test' / 'concat_data.csv'


    if args.method == 'svc':
        svc_classification(args, params, train_filepath, val_filepath, test_filepath)
    elif args.method == 'nn':
        nn_classification(args, params, train_filepath, val_filepath, test_filepath)

    '''
    # Get methods and hyperparameters grid
    classifier, param_grid = classification_methods.get_classifier_param_grid(args.method, params)
    balancing_sampler = class_balancing.get_balancing_method(args.balancing, params)
    estimators = [StandardScaler()]
    if args.balancing:
        estimators.append(class_balancing.get_balancing_method(args.balancing, params))
    estimators.extend(classifier)
    print(estimators)
    print(*estimators)

    # create pipeline
    if args.balancing:
        pipe = imb_make_pipeline(*estimators)
    else:
        pipe = sk_make_pipeline(*estimators)

    print(pipe)
    print(pipe.get_params())

    for par in tqdm(param_grid):
        print(f'fitting with {par}...')
        pipe.set_params(**par)
        pipe.fit(X_train, y_train)
        print('predicting...')
        y_pred = pipe.predict(X_val)
        val_score = params['scoring'](y_val, y_pred)
        print(f"£Score {params['scoring'].__name__}: {val_score}")
        break

    # for _, df_chunk in X_train.groupby(np.arange(len(X_train)) // batchsize):
    #     X_train_chunk = df_chunk.iloc[:, :-1]
    #     scaler.partial_fit(X_train_chunk)


    print('>> Balance classes, scale data and start training...')
    training_generator = BalancedBatchGenerator(X_train, y_train, sampler=balancing_sampler, batch_size=batchsize,
                                                random_state=params['random_state'])

    '''

'''
def main():
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    params = utils.read_config_file(args.cfg, args.method)

    # Read features from file
    # ---TODO FARE SCRIPT PER CONCATENARE FEATURES---
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

    # TODO FARE SCALER CON GENERATOR

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
'''


if __name__ == '__main__':
    main()