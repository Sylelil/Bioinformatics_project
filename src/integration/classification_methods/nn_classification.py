import math
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import paths
from src.integration import data_generator, utils
from src.common import plots, classification_report_utils
from src.integration.classification_methods import common
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn import metrics

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight):

    TN = tf.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    TP = tf.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 1)

    FP = tf.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    FN = tf.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)

    # Converted as Keras Tensors
    TN = K.sum(K.variable(TN))
    FP = K.sum(K.variable(FP))

    specificity = TN / (TN + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    return 1.0 - (recall_weight*recall + spec_weight*specificity)


# def custom_loss(recall_weight, spec_weight):
#
#     def recall_spec_loss(y_true, y_pred):
#         return binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight)
#
#     # Returns the (y_true, y_pred) loss function
#     return recall_spec_loss


def custom_loss(y_true, y_pred):
    print('custom loss-----------------')
    TN = tf.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    TP = tf.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 1)

    FP = tf.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    FN = tf.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)

    # Converted as Keras Tensors
    TN = tf.reduce_sum(tf.cast(TN, tf.int64)).astype('int64')
    # TN = tf.math.count_nonzero(TN)
    #TN = K.sum(K.variable(TN))
    print(TN)
    FP = tf.reduce_sum(tf.cast(FP, tf.int64)).astype('int64')
    # FP = tf.math.count_nonzero(FP)
    # FP = K.sum(K.variable(FP))
    print(FP)

    specificity = TN / (TN + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    recall_weight = 0.9
    spec_weight = 0.1
    return 1.0 - (recall_weight * recall + spec_weight * specificity)


def make_model(n_input_features, lr, units_1, units_2, metrics=common.METRICS_keras, output_bias=None):
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

    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                  loss=keras.losses.BinaryCrossentropy(),
                  # loss=custom_loss,
                  metrics=metrics,
                  # run_eagerly=True
                  )
    return model


def mpl_classify(X_train, y_train, X_val, y_val, X_test, y_test, mlp_settings, balancer=None, scaler=None, train_path=None,
                 use_generators=False):
    """
       Description: Train and test MLP classifier.
       :param X_train: train data.
       :param y_train: train labels.
       :param X_val: validation data.
       :param y_val: validation labels.
       :param X_test: test data.
       :param y_test: test labels.
       :param mlp_settings: dictionary with MLP settings.
       :balancer: balancing method.
       :scaler: scaling method.
       :train_path: path to training data.
       :param use_generators: either or not to get data as generators (default: False).
       :returns: y_pred_test, y_pred_train, test_scores, history: validation and test results
    """

    print(">> Creating MultiLayer Perceptron model...")
    model = make_model(mlp_settings['n_input_features'], mlp_settings['learning_rate'], units_1=mlp_settings['units_1'],
                       units_2=mlp_settings['units_2'])
    model.summary()

    print(">> Fitting model on train data...")
    print("num val steps: " + str(math.ceil(len(y_val) / mlp_settings['BATCH_SIZE'])))
    print("num train steps: " + str(math.ceil(len(y_train) / mlp_settings['BATCH_SIZE'])))
    print("num test steps: " + str(math.ceil(len(y_test) / mlp_settings['BATCH_SIZE'])))
    history = model.fit(x=X_train,
                        y=(None if use_generators else y_train),
                        steps_per_epoch=(math.ceil(len(y_train) / mlp_settings['BATCH_SIZE']) if use_generators else None),
                        batch_size=mlp_settings['BATCH_SIZE'],
                        epochs=mlp_settings['EPOCHS'],
                        callbacks=mlp_settings['early_stopping'],
                        validation_data=(X_val, (None if use_generators else y_val)),
                        validation_steps=((len(y_val) // mlp_settings['BATCH_SIZE']) + 1 if use_generators else None),
                        class_weight=mlp_settings['class_weight'],
                        verbose=1)

    if use_generators:
        X_train = data_generator.csv_data_generator(train_path,
                                                    batchsize=mlp_settings['BATCH_SIZE'],
                                                    scaler=scaler,
                                                    mode='eval',
                                                    balancer=balancer)
    print("Predict on train..")
    y_pred_train = model.predict(X_train, batch_size=mlp_settings['BATCH_SIZE'],
                                 steps=((len(y_train) // mlp_settings['BATCH_SIZE']) + 1 if use_generators else None)
                                 )
    print("Predict on test..")
    y_pred_test = model.predict(X_test, batch_size=mlp_settings['BATCH_SIZE'],
                                steps=((len(y_test) // mlp_settings['BATCH_SIZE']) + 1 if use_generators else None)
                                )
    print("Len test")
    print(len(y_pred_test))
    print(len(y_test))

    print("Len train")
    print(len(y_pred_train))
    print(len(y_train))

    print('Evaluating model on the test dataset...')
    test_scores_list = model.evaluate(X_test, (None if use_generators else y_test),
                                      batch_size=mlp_settings['BATCH_SIZE'],
                                      verbose=1,
                                      steps=((len(y_test) // mlp_settings['BATCH_SIZE']) + 1 if use_generators else None)
                                      )
    
    test_scores = dict(zip(model.metrics_names, test_scores_list))
    y_pred_test_labels = [1 if x > 0.5 else 0 for x in y_pred_test]
    test_scores['matthews_corrcoef'] = metrics.matthews_corrcoef(y_test, y_pred_test_labels)

    return y_pred_test, y_pred_train, test_scores, history


def pca_nn_classifier(args, params, train_filepath, val_filepath, test_filepath):
    """
       Description: Train and test MLP classifier preceded by IncrementalPCA and class balancing, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param train_filepath: train data path.
       :param val_filepath: validation data path.
       :param test_filepath: test data path.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = utils.compute_scaling_pca(params, train_filepath, val_filepath,
                                                                               test_filepath)
    train_len = len(y_train)
    val_len = len(y_val)
    test_len = len(y_test)
    print(f'>> train len = {train_len}')
    print(f'>> val len = {val_len}')
    print(f'>> test len = {test_len}')

    class_weight = None
    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = common.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)
    elif args.balancing == 'weights':
        class_weight = common.compute_class_weights(y_train)

    mlp_settings = {
        'n_input_features': params['pca']['n_components'],
        'EPOCHS': params['pca_nn']['epochs'],
        'BATCH_SIZE': params['pca_nn']['batchsize'],
        'early_stopping': tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True),
        'learning_rate': params['pca_nn']['lr'],
        'class_weight': class_weight,
        'units_1': params['pca_nn']['units_1'],
        'units_2': params['pca_nn']['units_2'],
    }

    y_pred_test, y_pred_train, test_scores, history = mpl_classify(X_train, y_train, X_val, y_val, X_test, y_test,
                                                                   mlp_settings)

    generate_classification_results(args, params, y_test, y_pred_test, y_train, y_pred_train, test_scores, history)


def nn_classifier(args, params, train_filepath, val_filepath, test_filepath):
    """
       Description: Train and test MLP classifier using data generators from csv files, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param train_filepath: train data path.
       :param val_filepath: validation data path.
       :param test_filepath: test data path.
    """
    batchsize = params['nn']['batchsize']

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

    train_len = len(y_train)
    val_len = len(y_val)
    test_len = len(y_test)
    print(f'>> train len = {train_len}')
    print(f'>> val len = {val_len}')
    print(f'>> test len = {test_len}')

    # get number of features
    with open(train_filepath, "r") as f:
        line = f.readline()
        line = line.strip().split(",")
        n_features = len(line) - 1  # -1 because we don't want to consider label!

    balancer = None
    class_weight = None
    if args.balancing and args.balancing != 'weights':
        balancer = common.get_balancing_method(args.balancing, params)
    elif args.balancing == 'weights':
        class_weight = common.compute_class_weights(y_train)

    print(f">> Creating train data generator with scaler and {args.balancing} balancer...")
    train_generator = data_generator.csv_data_generator(train_filepath,
                                                        batchsize=batchsize,
                                                        scaler=scaler,
                                                        mode='train',
                                                        balancer=balancer)
    print(f">> Creating validation data generator with scaler...")
    val_generator = data_generator.csv_data_generator(val_filepath,
                                                      batchsize=batchsize,
                                                      scaler=scaler,
                                                      mode='eval',
                                                      balancer=None)
    print(f">> Creating test data generator with scaler...")
    test_generator = data_generator.csv_data_generator(test_filepath,
                                                       batchsize=batchsize,
                                                       scaler=scaler,
                                                       mode='eval',
                                                       balancer=None)
    mlp_settings = {
        'n_input_features': n_features,
        'EPOCHS': params['nn']['epochs'],
        'BATCH_SIZE': batchsize,
        'early_stopping': tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True),
        'learning_rate': params['nn']['lr'],
        'class_weight': class_weight,
        'units_1': params['nn']['units_1'],
        'units_2': params['nn']['units_2'],
    }

    y_pred_test, y_pred_train, test_scores, history = mpl_classify(X_train=train_generator, y_train=y_train,
                                                                   X_val=val_generator, y_val=y_val,
                                                                   X_test=test_generator, y_test=y_test,
                                                                   mlp_settings=mlp_settings,
                                                                   balancer=balancer, scaler=scaler,
                                                                   train_path=train_filepath,
                                                                   use_generators=True)

    generate_classification_results(args, params, y_test, y_pred_test, y_train, y_pred_train, test_scores, history)


def generate_classification_results(args, params, y_test, y_pred_test, y_train, y_pred_train, test_scores, history):
    """
       Description: Generate classification report and plots.
       :param args: arguments.
       :param params: parameters.
       :param y_test: ground truth test labels.
       :param y_pred_test: predicted test labels.
       :param y_train: ground truth train labels.
       :param y_pred_train: predicted train labels.
       :param test_scores: dictionary with scores of test classification.
       :param history: model history.
    """
    # path to save results:
    experiment_descr = f"CLF_{args.classification_method}_PCA_{params['pca']['n_components']}_BAL_{args.balancing}"
    results_path = Path(paths.integration_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # convert predicted probabilities (output of sigmoid) to 0/1 labels:
    y_pred_test_labels = [1 if x > 0.5 else 0 for x in y_pred_test]
    y_pred_train_labels = [1 if x > 0.5 else 0 for x in y_pred_train]

    # generate classification report:
    experiment_info = {}
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['PCA n. components'] = str(params['pca']['n_components'])
    experiment_info['Class balancing method'] = str(args.balancing)
    classification_report_utils.generate_classification_report(results_path, y_test, y_pred_test_labels, test_scores, experiment_info)

    # generate plots:
    plots.plot_train_val_results(history, results_path)
    classification_report_utils.generate_classification_plots(results_path, y_test, y_pred_test_labels, y_train, y_pred_train_labels)

    print('>> Done')
