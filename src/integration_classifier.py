import os
import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
import sklearn
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from src.common import class_balancing, utils, data_generator, classification, plots
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.keras import BalancedBatchGenerator
from sklearn.pipeline import make_pipeline as sk_make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import IncrementalPCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


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


def compute_scaling_pca(params, train_filepath, val_filepath, test_filepath):
    x_train_pca_path = Path('assets') / 'concatenated_pca' / 'x_train.npy'
    y_train_pca_path = Path('assets') / 'concatenated_pca' / 'y_train.npy'
    x_val_pca_path = Path('assets') / 'concatenated_pca' / 'x_val.npy'
    y_val_pca_path = Path('assets') / 'concatenated_pca' / 'y_val.npy'
    x_test_pca_path = Path('assets') / 'concatenated_pca' / 'x_test.npy'
    y_test_pca_path = Path('assets') / 'concatenated_pca' / 'y_test.npy'

    if os.path.exists(Path('assets') / 'concatenated_pca'):
        print('>> Reading files with scaled and pca data previously computed...')
        X_train = np.load(x_train_pca_path)
        y_train = np.load(y_train_pca_path)
        X_val = np.load(x_val_pca_path)
        y_val = np.load(y_val_pca_path)
        X_test = np.load(x_test_pca_path)
        y_test = np.load(y_test_pca_path)

    else:
        os.mkdir(Path('assets') / 'concatenated_pca')
        batchsize = params['batchsize']

        print(">> Fitting scaler...")
        scaler = StandardScaler()
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            scaler.partial_fit(X_train_chunk)

        ipca = IncrementalPCA(n_components=params['n_components'])
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []
        print(">> Transforming train data with scaler and fitting incremental pca...")
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            X_train_chunk_scaled = scaler.transform(X_train_chunk)
            ipca.partial_fit(X_train_chunk_scaled)
        print(">> Transforming train data with incremental pca...")
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            y_train_chunk = chunk['label']
            X_train_chunk_scaled = scaler.transform(X_train_chunk)
            X_train_chunk_ipca = ipca.transform(X_train_chunk_scaled)
            X_train.extend(X_train_chunk_ipca)
            y_train.extend(y_train_chunk)
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

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        print(">> Saving computed features on files in assets/concatenated_pca/ folder...")
        np.save(x_train_pca_path, X_train)
        np.save(y_train_pca_path, y_train)
        np.save(x_val_pca_path, X_val)
        np.save(y_val_pca_path, y_val)
        np.save(x_test_pca_path, X_test)
        np.save(y_test_pca_path, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def shallow_classification(args, params, train_filepath, val_filepath, test_filepath):
    X_train, y_train, X_val, y_val, X_test, y_test = compute_scaling_pca(params, train_filepath, val_filepath, test_filepath)

    if args.balancing and args.balancing != 'weights':
            print(f">> Applying class balancing with {args.balancing}...")
            balancer = class_balancing.get_balancing_method(args.balancing, params)
            X_train, y_train = balancer.fit_resample(X_train, y_train)

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
                                   random_state=params['random_state'])
        else:
            classifier = SGDClassifier(alpha=hyperparam,
                                       max_iter=10, # np.ceil(10**6 / n_samples)
                                       class_weight=('balanced' if args.balancing == 'weights' else None),
                                       random_state=params['random_state'])
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        score = params['scoring'](y_val, y_pred)
        print(f"    Validation {params['scoring'].__name__}: {score}")
        if score > best_score:
            best_score = score
            best_hyperparam = hyperparam

    print(f"Best {params['scoring'].__name__} ({'C' if args.method == 'svc' else 'alpha'}={best_hyperparam}): {best_score}")
    print(f">> Training with best {'LinearSVC' if args.method == 'svc' else 'SGDClassifier'} model...")
    if args.method == 'svc':
        best_classifier = LinearSVC(C=best_hyperparam,
                                    class_weight=('balanced' if args.balancing == 'weights' else None),
                                    random_state=params['random_state'])
    else:
        best_classifier = SGDClassifier(alpha=best_hyperparam,
                                        max_iter=10,
                                        class_weight=('balanced' if args.balancing == 'weights' else None),
                                        random_state=params['random_state'])
    best_classifier.fit(X_train, y_train)
    print(">> Testing...")
    y_pred_test = best_classifier.predict(X_test)
    test_score = params['scoring'](y_test, y_pred_test)
    print(f"Test {params['scoring'].__name__} = {test_score}")

    print(classification_report(y_test, y_pred_test))
    print('>> Done')


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def make_model(n_input_features, units_1, units_2, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(units_1, activation='relu', input_shape=(n_input_features,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units_2, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=metrics)
    return model

def compute_class_weights(labels):
    print(">> Computing class weights...")
    total = len(labels)
    pos = np.count_nonzero(labels)
    neg = total - pos
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * total / 2.0
    weight_for_1 = (1 / pos) * total / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight


def pca_nn_classification(args, params, train_filepath, val_filepath, test_filepath):
    X_train, y_train, X_val, y_val, X_test, y_test = compute_scaling_pca(params, train_filepath, val_filepath, test_filepath)

    class_weight = None
    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = class_balancing.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)
    elif args.balancing == 'weights':
        class_weight = compute_class_weights(y_train)

    mlp_settings = {
        'n_input_features' : params['n_components'],
        'EPOCHS' : params['epochs'],
        'BATCH_SIZE' : params['batchsize_nn'],
        'early_stopping' : tf.keras.callbacks.EarlyStopping(monitor='val_prc',
                                                            verbose=1,
                                                            patience=10,
                                                            mode='max',
                                                            restore_best_weights=True),
        'class_weight' : class_weight,
        'units_1' : 32,
        'units_2': 16,
    }

    mpl_classify(X_train, y_train, X_val, y_val, X_test, y_test, mlp_settings)


def mpl_classify(X_train, y_train, X_val, y_val, X_test, y_test, mlp_settings, use_generators=False):

    print(">> Creating MultiLayer Perceptron model...")
    model = make_model(mlp_settings['n_input_features'], units_1=mlp_settings['units_1'], units_2=mlp_settings['units_2'])
    model.summary()

    print(">> Fitting model on train data...")
    history = model.fit(x=X_train,
                        y=(None if use_generators else y_train),
                        batch_size=mlp_settings['BATCH_SIZE'],
                        epochs=mlp_settings['EPOCHS'],
                        callbacks=mlp_settings['early_stopping'],
                        validation_data=(X_val, (None if use_generators else y_val)),
                        class_weight=mlp_settings['class_weight'],
                        verbose=1)


    plots.plot_loss(history, 'Training and validation loss', 0)
    plots.plt.figure()
    plots.plot_metrics(history)
    plt.figure()

    train_predictions_baseline = model.predict(X_train, batch_size=mlp_settings['BATCH_SIZE'])
    test_predictions_baseline = model.predict(X_test, batch_size=mlp_settings['BATCH_SIZE'])

    print('Evaluating model on the test dataset...')
    baseline_results = model.evaluate(X_test, (None if use_generators else y_test), batch_size=mlp_settings['BATCH_SIZE'], verbose=0)

    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    plots.plot_cm(y_test, test_predictions_baseline)
    plt.figure()

    plots.plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
    plots.plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.figure()

    plots.plot_prc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
    plots.plot_prc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.figure()

    plt.show()


def nn_classification(args, params, train_filepath, val_filepath, test_filepath):
    batchsize = params['batchsize_nn']

    scaler = StandardScaler()

    print(">> Fitting scaler on train data and extracting labels...")
    if os.path.exists(Path('assets') / 'concatenated_pca'):
        y_train_path = Path('assets') / 'concatenated_pca' / 'y_train.npy'
        y_val_path = Path('assets') / 'concatenated_pca' / 'y_val.npy'
        y_test_path = Path('assets') / 'concatenated_pca' / 'y_test.npy'
        y_train = np.load(y_train_path)
        y_val = np.load(y_val_path)
        y_test = np.load(y_test_path)
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            scaler.partial_fit(X_train_chunk)
    else:
        y_train = []
        y_val = []
        y_test = []
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            scaler.partial_fit(X_train_chunk)
            y_train_chunk = chunk['label']
            y_train.extend(y_train_chunk)
        for chunk in tqdm(pd.read_csv(val_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            y_val_chunk = chunk['label']
            y_val.extend(y_val_chunk)
        for chunk in tqdm(pd.read_csv(test_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            y_test_chunk = chunk['label']
            y_test.extend(y_test_chunk)

    # get number of features
    with open(train_filepath, "r") as f:
        line = f.readline()
        line = line.strip().split(",")
        n_features = len(line) -1  # -1 because we don't want to consider label!

    balancer = None
    class_weight = None
    if args.balancing and args.balancing != 'weights':
        balancer = class_balancing.get_balancing_method(args.balancing, params)
    elif args.balancing == 'weights':
        class_weight = compute_class_weights(y_train)

    print(f">> Creating train data generator with scaler and {args.balancing} balancer...")
    train_generator = data_generator.csv_data_generator(train_filepath,
                                                        batchsize=batchsize,
                                                        scaler=scaler,
                                                        balancer=balancer)
    print(f">> Creating validation data generator with scaler...")
    val_generator = data_generator.csv_data_generator(val_filepath,
                                                        batchsize=batchsize,
                                                        scaler=scaler,
                                                        balancer=None)
    print(f">> Creating test data generator with scaler...")
    test_generator = data_generator.csv_data_generator(test_filepath,
                                                        batchsize=batchsize,
                                                        scaler=scaler,
                                                        balancer=None)
    mlp_settings = {
        'n_input_features' : n_features,
        'EPOCHS' : params['epochs'],
        'BATCH_SIZE' : batchsize,
        'early_stopping' : tf.keras.callbacks.EarlyStopping(monitor='val_prc',
                                                            verbose=1,
                                                            patience=10,
                                                            mode='max',
                                                            restore_best_weights=True),
        'class_weight' : class_weight,
        'units_1' : 2048,
        'units_2': 1024,
    }

    mpl_classify(X_train=train_generator, y_train=y_train,
                 X_val=val_generator, y_val=y_val,
                 X_test=test_generator, y_test=y_test,
                 mlp_settings=mlp_settings,
                 use_generators=True)



def main():
    # Parse arguments from command line
    args = args_parse()

    # Read configuration file
    params = utils.read_config_file(args.cfg)

    train_filepath = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'train' / 'concat_data.csv'
    val_filepath = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'val' / 'concat_data.csv'
    test_filepath = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'test' / 'concat_data.csv'

    train_filepath_copied_genes = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'train' / 'concat_data_copied.csv'
    val_filepath_copied_genes = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'val' / 'concat_data_copied.csv'
    test_filepath_copied_genes = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated' / 'test' / 'concat_data_copied.csv'

    if args.method == 'svc' or args.method == 'sgd':
        shallow_classification(args, params, train_filepath, val_filepath, test_filepath)
    elif args.method == 'pca_nn':
        pca_nn_classification(args, params, train_filepath, val_filepath, test_filepath)
    elif args.method == 'nn':
        nn_classification(args, params, train_filepath_copied_genes, val_filepath_copied_genes, test_filepath_copied_genes)

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