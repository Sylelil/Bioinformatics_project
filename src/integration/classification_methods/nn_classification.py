import math
import os

from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import paths
from src.common.classification_metrics import METRICS_keras, top_metric_keras
from src.integration import data_generator, utils
from src.common import plots, classification_report_utils
from src.integration.classification_methods import common
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def make_model(n_input_features, lr, units_1, units_2, metrics=METRICS_keras, output_bias=None):
    """
       Description: Create MLP model.
       :param n_input_features: number of input features.
       :param lr: learning rate.
       :param units_1: number of units in first dense layer.
       :param units_2: number of units in second dense layer.
       :param metrics: metrics list.
       :param output_bias: output bias (default: None).
       :returns: MLP model.
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_input_features,)),
        keras.layers.Dense(units_1, activation='relu', ),
        keras.layers.Dropout(0.5),
        # keras.layers.Dense(units_2, activation='relu'),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.BinaryCrossentropy(),
                  # loss=custom_loss,
                  metrics=metrics,
                  # run_eagerly=True
                  )
    return model


def mlp_classify(X_train, y_train, X_test, y_test, mlp_settings, use_generators=False):
    model = make_model(mlp_settings['n_input_features'], mlp_settings['learning_rate'], units_1=mlp_settings['units_1'],
                       units_2=mlp_settings['units_2'])
    model.summary()
    print("num train steps: " + str(len(y_train) // mlp_settings['BATCH_SIZE']))
    print("num val steps: " + str(len(y_test) // mlp_settings['BATCH_SIZE'] + 1))
    history = model.fit(x=X_train,
                        y=(None if use_generators else y_train),
                        steps_per_epoch=(len(y_train) // mlp_settings['BATCH_SIZE'] if use_generators else None),
                        batch_size=mlp_settings['BATCH_SIZE'],
                        epochs=mlp_settings['EPOCHS'],
                        callbacks=mlp_settings['early_stopping'],
                        validation_data=(X_test, (None if use_generators else y_test)),
                        validation_steps=(
                            (len(y_test) // mlp_settings['BATCH_SIZE'] + 1) if use_generators else None),
                        class_weight=mlp_settings['class_weight'],
                        verbose=1)

    print("Predict on train..")
    y_pred_train = model.predict(X_train, batch_size=mlp_settings['BATCH_SIZE'],
                                 steps=(
                                     (len(y_train) // mlp_settings['BATCH_SIZE']) + 1 if use_generators else None)
                                 )
    y_pred_train_labels = [1 if x > 0.5 else 0 for x in y_pred_train]
    print("Predict on test..")
    y_pred_test = model.predict(X_test, batch_size=mlp_settings['BATCH_SIZE'],
                                steps=((len(y_test) // mlp_settings['BATCH_SIZE']) + 1 if use_generators else None)
                                )
    y_pred_test_labels = [1 if x > 0.5 else 0 for x in y_pred_test]

    print('Evaluating model on test...')
    test_scores_list = model.evaluate(X_test, (None if use_generators else y_test),
                                      batch_size=mlp_settings['BATCH_SIZE'],
                                      verbose=1,
                                      steps=(
                                          (len(y_test) // mlp_settings[
                                              'BATCH_SIZE']) + 1 if use_generators else None)
                                      )
    test_scores = dict(zip(model.metrics_names, test_scores_list))
    test_scores['matthews_corrcoef'] = metrics.matthews_corrcoef(y_test, y_pred_test_labels)
    print('Evaluating model on train...')
    train_scores_list = model.evaluate(X_train, (None if use_generators else y_train),
                                       batch_size=mlp_settings['BATCH_SIZE'],
                                       verbose=1,
                                       steps=(
                                           (len(y_train) // mlp_settings[
                                               'BATCH_SIZE']) + 1 if use_generators else None)
                                       )
    train_scores = dict(zip(model.metrics_names, train_scores_list))
    train_scores['matthews_corrcoef'] = metrics.matthews_corrcoef(y_train, y_pred_train_labels)

    return y_pred_train, y_pred_test, train_scores, test_scores, history


def mlp_cross_validate(X_train, y_train, X_test, y_test, mlp_settings, params):
    """
       Description: Train and test MLP classifier.
       :param X_train: train data.
       :param y_train: train labels.
       :param X_test: test data.
       :param y_test: test labels.
       :param mlp_settings: dictionary with MLP settings.
       :param data_path: path to data.
       :param use_generators: either or not to get data as generators (default: False).
       :returns: y_pred_test, y_pred_train, test_scores, history: validation and test results
    """
    metric_name = top_metric_keras

    print(">> Creating MultiLayer Perceptron model...")
    # model = KerasClassifier(build_fn=make_model(mlp_settings['n_input_features'],
    #                                             mlp_settings['learning_rate'],
    #                                             units_1=mlp_settings['units_1'],
    #                                             units_2=mlp_settings['units_2']),
    #                         epochs=mlp_settings['EPOCHS'],
    #                         batch_size=mlp_settings['BATCH_SIZE'],
    #                         verbose=1)

    # cross validation for unbiased error estimation
    test_results = []
    train_results = []

    cv = KFold(n_splits=params['cv']['n_inner_splits'], shuffle=True,
               random_state=params['general']['random_state'])

    for train_ix, test_ix in tqdm(cv.split(X_train, y_train)):
        # split data in k folds
        X_train_cv, X_test_cv = X_train[train_ix, :], X_train[test_ix, :]
        y_train_cv, y_test_cv = y_train[train_ix], y_train[test_ix]

        y_pred_train_cv, y_pred_test_cv, train_scores_cv, test_scores_cv, _ = mlp_classify(X_train_cv, y_train_cv,
                                                                                           X_test_cv, y_test_cv,
                                                                                           mlp_settings, params)

        test_results.append(test_scores_cv[metric_name])
        train_results.append(train_scores_cv[metric_name])

        print('CV:')
        print(f"    Val recall = {test_scores_cv[metric_name]} - ", end='')
        print(f"Val precision = {test_scores_cv['precision']} - ", end='')
        print(f"Val accuracy = {test_scores_cv['accuracy']} - ", end='')
        print(f"Val matthews_corrcoef = {test_scores_cv['matthews_corrcoef']} - ", end='')
        print(f"Train recall = {train_scores_cv[metric_name]} - ", end='')
        print(f"Train precision = {train_scores_cv['precision']} - ", end='')
        print(f"Train accuracy = {train_scores_cv['accuracy']} - ", end='')
        print(f"Train matthews_corrcoef = {train_scores_cv['matthews_corrcoef']}")
        print()

    # calculate the mean score over all K folds, and report as the generalization error
    global_test_score = np.mean(test_results)
    global_test_std = np.std(test_results)
    global_train_score = np.mean(train_results)
    global_train_std = np.std(train_results)
    print()
    print(f"Global validation {metric_name} = {str(global_test_score)} ({str(global_test_std)})")
    print(f"Global training {metric_name} = {str(global_train_score)} ({str(global_train_std)})")
    print()

    # refit on train data to evaluate on test data
    print('>> Fitting on train dataset to evaluate on test dataset...')
    y_pred_train, y_pred_test, train_scores, test_scores, history = mlp_classify(X_train, y_train,
                                                                                 X_test, y_test,
                                                                                 mlp_settings, params)

    results = {
        'metric_name': metric_name,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train,
        'test_scores': test_scores,
        'history': history,
        'global_test_score': global_test_score,
        'global_test_std': global_test_std,
        'global_train_score': global_train_score,
        'global_train_std': global_train_std,
    }
    return results


def nn_classifier(args, params, data_path):
    """
       Description: Train and test MLP classifier preceded by IncrementalPCA and class balancing, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param data_path: data path.
       :param n_features_images: number of features of images to be considered
    """
    X_train, y_train, X_test, y_test = utils.get_data(data_path)

    train_len = len(y_train)
    test_len = len(y_test)
    print(f'>> train len = {train_len}')
    print(f'>> test len = {test_len}')

    # get number of features
    with open(Path(data_path) / 'x_train.csv', "r") as f:
        line = f.readline()
        line = line.strip().split(",")
        n_features = len(line)
    print(f'>> n. features = {n_features}')

    # get class weight to do class balancing
    class_weight = common.compute_class_weights(y_train)

    mlp_settings = {
        'n_input_features': n_features,
        'EPOCHS': params['epochs'],
        'BATCH_SIZE': params['batchsize'],
        'early_stopping': tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True),
        'learning_rate': params['lr'],
        'class_weight': class_weight,
        'units_1': params['units_1'],
        'units_2': params['units_2'],
    }

    results = mlp_cross_validate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                 mlp_settings=mlp_settings, params=params)

    generate_classification_results(args, params, y_test, y_train, data_path, results)


def generate_classification_results(args, params, y_test, y_train, data_path, results):
    """
       Description: Generate classification report and plots.
       :param params: parameters.
       :param args: arguments.
       :param y_test: ground truth test labels.
       :param y_pred_test: predicted test labels.
       :param y_train: ground truth train labels.
       :param y_pred_train: predicted train labels.
       :param test_scores: dictionary with scores of test classification.
       :param history: model history.
       :param data_path: data path.
    """
    # path to save results:
    experiment_descr = f"{os.path.split(data_path)[1]}"
    experiment_descr += f"_{params['general']['use_features_images_only']}"
    experiment_descr += f"_{args.classification_method}+{params[args.classification_method]['lr']}+{params[args.classification_method]['epochs']}+{params[args.classification_method]['units_1']}"
    experiment_descr += f"_weigths"
    results_path = Path(paths.integration_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    metric_name = results['metric_name']
    test_scores = results['test_scores']
    y_pred_test = results['y_pred_test']
    y_pred_train = results['y_pred_train']
    history = results['history']
    global_test_std = results['global_test_std']
    global_test_score = results['global_test_score']
    global_train_score = results['global_train_score']
    global_train_std = results['global_train_std']

    # convert predicted probabilities (output of sigmoid) to 0/1 labels:
    y_pred_test_labels = [1 if x > 0.5 else 0 for x in y_pred_test]
    y_pred_train_labels = [1 if x > 0.5 else 0 for x in y_pred_train]
    pred_proba_test = y_pred_test
    pred_proba_train = y_pred_train

    # generate classification report:
    experiment_info = {}
    experiment_info['Data folder'] = str(data_path)
    experiment_info['Selected features'] = f'Image features, no gene features' if params['general'][
        'use_features_images_only'] else 'All'
    experiment_info['PCA'] = f"n. components={params['general']['num_principal_components']}" if params['general'][
        'num_principal_components'] else 'No'
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['Learning rate'] = params[args.classification_method]['lr']
    experiment_info['N. epochs'] = params[args.classification_method]['epochs']
    experiment_info['N. units hidden layer'] = params[args.classification_method]['units_1']
    experiment_info['Class balancing method'] = 'weights'
    experiment_info['Error estimation:'] = '---------------------------------------'
    experiment_info['Global validation score'] = f"{metric_name}={str(global_test_score)} ({str(global_test_std)})"
    experiment_info['Global train score'] = f"{metric_name}={str(global_train_score)} ({str(global_train_std)})"
    experiment_info['Test results:'] = '---------------------------------------'
    experiment_info['Final test score'] = f"{metric_name}={test_scores[metric_name]}"

    test_data_info_path = data_path / 'info_test.csv'
    classification_report_utils.generate_classification_report(results_path, y_test, y_pred_test_labels, test_scores,
                                                               experiment_info, test_data_info_path=test_data_info_path)

    # generate plots:
    plots.plot_train_val_results(history, results_path)
    classification_report_utils.generate_classification_plots_nn(results_path, y_test, y_pred_test_labels,
                                                                 pred_proba_test, y_train, y_pred_train_labels,
                                                                 pred_proba_train)

    print('>> Done')
