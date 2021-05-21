import os
import pandas as pd
from pathlib import Path
from config import paths
from common.classification_metrics import METRICS_keras
from integration import data_generator, utils
from common import plots, classification_report_utils
from integration.classification_methods import common
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

    return 1.0 - (recall_weight * recall + spec_weight * specificity)


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
    # TN = K.sum(K.variable(TN))
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


def mpl_classify(X_train, y_train, X_val, y_val, X_test, y_test, mlp_settings,
                 n_features_images, balancer=None, data_path=None, use_generators=False):
    """
       Description: Train and test MLP classifier.
       :param n_features_images: number of features of images to be considered.
       :param X_train: train data.
       :param y_train: train labels.
       :param X_val: validation data.
       :param y_val: validation labels.
       :param X_test: test data.
       :param y_test: test labels.
       :param mlp_settings: dictionary with MLP settings.
       :param balancer: balancing method.
       :param scaler: scaling method.
       :param data_path: path to data.
       :param use_generators: either or not to get data as generators (default: False).
       :returns: y_pred_test, y_pred_train, test_scores, history: validation and test results
    """

    print(">> Creating MultiLayer Perceptron model...")
    model = make_model(mlp_settings['n_input_features'], mlp_settings['learning_rate'], units_1=mlp_settings['units_1'],
                       units_2=mlp_settings['units_2'])
    model.summary()

    print(">> Fitting model on train data...")
    print("num val steps: " + str(len(y_val) // mlp_settings['BATCH_SIZE'] + 1))
    print("num train steps: " + str(len(y_train) // mlp_settings['BATCH_SIZE']))
    print("num test steps: " + str(len(y_test) // mlp_settings['BATCH_SIZE'] + 1))
    history = model.fit(x=X_train,
                        y=(None if use_generators else y_train),
                        steps_per_epoch=(len(y_train) // mlp_settings['BATCH_SIZE'] if use_generators else None),
                        batch_size=mlp_settings['BATCH_SIZE'],
                        epochs=mlp_settings['EPOCHS'],
                        callbacks=mlp_settings['early_stopping'],
                        validation_data=(X_val, (None if use_generators else y_val)),
                        validation_steps=((len(y_val) // mlp_settings['BATCH_SIZE'] + 1) if use_generators else None),
                        class_weight=mlp_settings['class_weight'],
                        verbose=1)

    if use_generators:
        X_train = data_generator.csv_data_generator(data_path,
                                                    batchsize=mlp_settings['BATCH_SIZE'],
                                                    n_features_images=n_features_images,
                                                    balancer=balancer,
                                                    dataset_name='train',
                                                    mode='eval')

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

    if use_generators:
        X_test = data_generator.csv_data_generator(data_path,
                                                   batchsize=mlp_settings['BATCH_SIZE'],
                                                   n_features_images=n_features_images,
                                                   balancer=None,
                                                   dataset_name='test',
                                                   mode='eval')

    print('Evaluating model on the test dataset...')
    test_scores_list = model.evaluate(X_test, (None if use_generators else y_test),
                                      batch_size=mlp_settings['BATCH_SIZE'],
                                      verbose=1,
                                      steps=(
                                          (len(y_test) // mlp_settings['BATCH_SIZE']) + 1 if use_generators else None)
                                      )

    test_scores = dict(zip(model.metrics_names, test_scores_list))
    y_pred_test_labels = [1 if x > 0.5 else 0 for x in y_pred_test]
    test_scores['matthews_corrcoef'] = metrics.matthews_corrcoef(y_test, y_pred_test_labels)

    return y_pred_test, y_pred_train, test_scores, history


def pca_nn_classifier(args, params, data_path, n_features_images):
    """
       Description: Train and test MLP classifier preceded by IncrementalPCA and class balancing, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param data_path: data path.
       :param n_features_images: number of features of images to be considered
    """
    X_train, y_train, X_val, y_val, X_test, y_test = utils.get_concatenated_data(data_path, n_features_images)

    train_len = len(y_train)
    val_len = len(y_val)
    test_len = len(y_test)
    print(f'>> train len = {train_len}')
    print(f'>> val len = {val_len}')
    print(f'>> test len = {test_len}')

    # get number of features
    if n_features_images:
        n_features = n_features_images
    else:
        with open(Path(data_path) / 'x_train.csv', "r") as f:
            line = f.readline()
            line = line.strip().split(",")
            n_features = len(line)
    print(f'>> n. features = {n_features}')

    class_weight = None
    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = common.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)
    elif args.balancing == 'weights':
        class_weight = common.compute_class_weights(y_train)

    mlp_settings = {
        'n_input_features': n_features,
        'EPOCHS': params['pcann']['epochs'],
        'BATCH_SIZE': params['pcann']['batchsize'],
        'early_stopping': tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True),
        'learning_rate': params['pcann']['lr'],
        'class_weight': class_weight,
        'units_1': params['pcann']['units_1'],
        'units_2': params['pcann']['units_2'],
    }

    y_pred_test, y_pred_train, test_scores, history = mpl_classify(X_train=X_train, y_train=y_train,
                                                                   X_val=X_val, y_val=y_val,
                                                                   X_test=X_test, y_test=y_test,
                                                                   mlp_settings=mlp_settings,
                                                                   n_features_images=n_features_images)

    generate_classification_results(args, params, y_test, y_pred_test, y_train, y_pred_train, test_scores, history, data_path)


def nn_classifier(args, params, data_path, n_features_images, use_generator=False):
    """
       Description: Train and test MLP classifier using data generators from csv files, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param data_path: data path.
       :param n_features_images: number of features of images to be considered
    """
    batchsize = params['nn']['batchsize']
    X_train = []
    X_val = []
    X_test = []

    if use_generator:

        y_train = pd.read_csv(Path(data_path) / 'y_train.csv', delimiter=',', header=None).values.ravel().astype(int)
        y_val = pd.read_csv(Path(data_path) / 'y_val.csv', delimiter=',', header=None).values.ravel().astype(int)
        y_test = pd.read_csv(Path(data_path) / 'y_test.csv', delimiter=',', header=None).values.ravel().astype(int)

    else:
        X_train, y_train, X_val, y_val, X_test, y_test = utils.get_concatenated_data(data_path, n_features_images)

    train_len = len(y_train)
    val_len = len(y_val)
    test_len = len(y_test)
    print(f'>> train len = {train_len}')
    print(f'>> val len = {val_len}')
    print(f'>> test len = {test_len}')

    # get number of features
    if n_features_images:
        n_features = n_features_images
    else:
        with open(Path(data_path) / 'x_train.csv', "r") as f:
            line = f.readline()
            line = line.strip().split(",")
            n_features = len(line)
    print(f'>> n. features = {n_features}')

    # get balancing method
    balancer = None
    class_weight = None
    if args.balancing and args.balancing != 'weights':
        balancer = common.get_balancing_method(args.balancing, params)
    elif args.balancing == 'weights':
        class_weight = common.compute_class_weights(y_train)

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

    if use_generator:
        print(f">> Creating train data generator with {args.balancing} balancer...")
        train_generator = data_generator.csv_data_generator(data_path,
                                                            batchsize=batchsize,
                                                            n_features_images=n_features_images,
                                                            balancer=balancer,
                                                            dataset_name='train',
                                                            mode='train')
        print(f">> Creating validation data generator...")
        val_generator = data_generator.csv_data_generator(data_path,
                                                          batchsize=batchsize,
                                                          n_features_images=n_features_images,
                                                          balancer=None,
                                                          dataset_name='val',
                                                          mode='eval')
        print(f">> Creating test data generator...")
        test_generator = data_generator.csv_data_generator(data_path,
                                                           batchsize=batchsize,
                                                           n_features_images=n_features_images,
                                                           balancer=None,
                                                           dataset_name='test',
                                                           mode='eval')

        y_pred_test, y_pred_train, test_scores, history = mpl_classify(X_train=train_generator, y_train=y_train,
                                                                       X_val=val_generator, y_val=y_val,
                                                                       X_test=test_generator, y_test=y_test,
                                                                       mlp_settings=mlp_settings,
                                                                       n_features_images=n_features_images,
                                                                       balancer=balancer,
                                                                       data_path=data_path,
                                                                       use_generators=True)
    else:

        y_pred_test, y_pred_train, test_scores, history = mpl_classify(X_train=X_train, y_train=y_train,
                                                                       X_val=X_val, y_val=y_val,
                                                                       X_test=X_test, y_test=y_test,
                                                                       mlp_settings=mlp_settings,
                                                                       n_features_images=n_features_images)

    generate_classification_results(args, params, y_test, y_pred_test, y_train, y_pred_train, test_scores, history, data_path)


def generate_classification_results(args, params, y_test, y_pred_test, y_train, y_pred_train, test_scores, history, data_path):
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
    experiment_descr += f"_{params['general']['n_features_images']}"
    experiment_descr += f"_{args.classification_method}+{params[args.classification_method]['lr']}+{params[args.classification_method]['epochs']}+{params[args.classification_method]['units_1']}"
    experiment_descr += f"_{args.balancing}"
    results_path = Path(paths.integration_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # convert predicted probabilities (output of sigmoid) to 0/1 labels:
    y_pred_test_labels = [1 if x > 0.5 else 0 for x in y_pred_test]
    y_pred_train_labels = [1 if x > 0.5 else 0 for x in y_pred_train]
    pred_proba_test = y_pred_test
    pred_proba_train = y_pred_train

    # generate classification report:
    experiment_info = {}
    experiment_info['Data folder'] = str(data_path)
    experiment_info['Selected features'] = f"{params['general']['n_features_images']} image features, no gene features" if params['general']['n_features_images'] else 'All'
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['Learning rate'] = params[args.classification_method]['lr']
    experiment_info['N. epochs'] = params[args.classification_method]['epochs']
    experiment_info['N. units hidden layer'] = params[args.classification_method]['units_1']
    experiment_info['Class balancing method'] = str(args.balancing)

    test_data_info_path = data_path / 'info_test.csv'
    classification_report_utils.generate_classification_report(results_path, y_test, y_pred_test_labels, test_scores,
                                                               experiment_info, test_data_info_path=test_data_info_path)

    # generate plots:
    plots.plot_train_val_results(history, results_path)
    classification_report_utils.generate_classification_plots_nn(results_path, y_test, y_pred_test_labels, pred_proba_test, y_train, y_pred_train_labels, pred_proba_train)

    print('>> Done')
