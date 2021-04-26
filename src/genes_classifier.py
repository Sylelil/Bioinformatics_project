import argparse
import os
import sys
from collections import Counter
from os import path
from imblearn.metrics import classification_report_imbalanced, sensitivity_score, specificity_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, \
    plot_confusion_matrix, plot_roc_curve, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import paths
from genes import methods
from genes.nn_classifier import MultiLayerPerceptron
import numpy as np


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)

    parser.add_argument('--classification_method',
                        help='Method to classify patients according to gene expression values',
                        choices=['svm', 'nn'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    if not os.path.exists(paths.svm_t_rfe_selected_features_train) or \
            len(os.listdir(paths.svm_t_rfe_selected_features_train)) == 0:
        print("Directory " + str(paths.svm_t_rfe_selected_features_train) + " doesn't exists or is empty")
        exit(1)
    if not os.listdir(paths.svm_t_rfe_selected_features_test) or \
            len(os.listdir(paths.svm_t_rfe_selected_features_test)) == 0:
        print("Directory " + str(paths.svm_t_rfe_selected_features_test) + " doesn't exists or is empty")
        exit(1)

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.classification_method)

    X_train, y_train = methods.load_selected_genes(paths.svm_t_rfe_selected_features_train)
    X_test, y_test = methods.load_selected_genes(paths.svm_t_rfe_selected_features_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    methods.pca(X_train, y_train)
    methods.pca(X_test, y_test)

    # SMOTE
    print("\n[SMOTE]")
    sm = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'])
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(Counter(y_train_sm))

    if args.classification_method == "svm":

        tuned_parameters = dict(svm__C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100])

        pipe_grid = Pipeline([('svm', SVC(kernel='linear'))])
        cv = KFold(n_splits=params['cv_grid_search_acc'])
        clf = GridSearchCV(estimator=pipe_grid, param_grid=tuned_parameters, scoring='accuracy', cv=cv, n_jobs=-1,
                           refit=True)
        clf.fit(X_train_sm, y_train_sm)
        pred = clf.predict(X_test)

        print("accuracy= %f" % accuracy_score(y_test, pred))
        average_precision = average_precision_score(y_test, pred, pos_label=0)
        precision = precision_score(y_test, pred, average='binary', pos_label=0)
        recall = recall_score(y_test, pred, average='binary', pos_label=0)
        sensitivity = sensitivity_score(y_test, pred, average='binary', pos_label=0)
        specificity = specificity_score(y_test, pred, average='binary', pos_label=0)
        print("\npos_label = 0")
        print('Average precision-recall score: %f' % average_precision)
        print('Precision score: %f' % precision)
        print('Recall score: %f' % recall)
        print('sensitivity: %f' % sensitivity)
        print('specificity: %f' % specificity)

        average_precision_1 = average_precision_score(y_test, pred, pos_label=1)
        precision_1 = precision_score(y_test, pred, average='binary', pos_label=1)
        recall_1 = recall_score(y_test, pred, average='binary', pos_label=1)
        sensitivity_1 = sensitivity_score(y_test, pred, average='binary', pos_label=1)
        specificity_1 = specificity_score(y_test, pred, average='binary', pos_label=1)
        print("\npos_label = 1")
        print('Average precision-recall score: %f' % average_precision_1)
        print('Precision score: %f' % precision_1)
        print('Recall score: %f' % recall_1)
        print('sensitivity: %f' % sensitivity_1)
        print('specificity: %f' % specificity_1)

        print(classification_report_imbalanced(y_test, pred))
        plot_confusion_matrix(clf, X_test, y_test)
        plt.show()
        plot_roc_curve(clf, X_test, y_test)
        plt.show()

        # Show decision boundary
        methods.show_svm_decision_boundary(clf, X_train_sm, y_train_sm, X_test, y_test)

    elif args.classification_method == "nn":

        n_epochs = 20
        my_net = MultiLayerPerceptron(n_epochs=n_epochs,
                                      batch_size=64,
                                      learning_rate=0.01,
                                      num_features=len(X_train[0]),
                                      n_classes=len(np.unique(y_test)),
                                      lr_reduction_epoch=15,
                                      logdir="fc_0-001")

        history = my_net.train_model(X_train_sm, y_train_sm, X_test, y_test)
        loss, acc = my_net.model.evaluate(X_test, y_test, verbose=False)
        print("Test done!\n\tMean accuracy: {}\n\tLoss: {}".format(acc, loss))

    else:
        sys.stderr.write("Invalid value for <classification_method>")
        exit(1)


if __name__ == "__main__":
    main()
