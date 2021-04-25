import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.common import class_balancing, data_generator, plots, classification_preprocessing, utils
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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


def pca_nn_classifier(args, params, train_filepath, val_filepath, test_filepath):
    X_train, y_train, X_val, y_val, X_test, y_test = classification_preprocessing.compute_scaling_pca(params, train_filepath, val_filepath, test_filepath)

    class_weight = None
    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = class_balancing.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)
    elif args.balancing == 'weights':
        class_weight = utils.compute_class_weights(y_train)

    mlp_settings = {
        'n_input_features' : params['pca']['n_components'],
        'EPOCHS' : params['nn']['epochs'],
        'BATCH_SIZE' : params['nn']['batchsize'],
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


def nn_classifier(args, params, train_filepath, val_filepath, test_filepath):
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
        class_weight = utils.compute_class_weights(y_train)

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
        'EPOCHS' : params['nn']['epochs'],
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

