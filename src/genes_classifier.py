import argparse
import math
import os
import sys
from collections import Counter
from os import path
from pathlib import Path
import seaborn as sns
from imblearn.metrics import classification_report_imbalanced, sensitivity_score, specificity_score
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap

from sklearn.metrics import accuracy_score, precision_score, average_precision_score, recall_score, \
    plot_confusion_matrix, plot_roc_curve, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

import matplotlib.pyplot as plt
from genes import methods


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

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.classification_method)

    if args.classification_method == "svm":
        training_selected_genes_dir = Path('results') / 'genes' / 'svm_t_rfe' / 'selected_features' / 'train'
        test_selected_genes_dir = Path('results') / 'genes' / 'svm_t_rfe' / 'selected_features' / 'test'
        if not os.listdir(training_selected_genes_dir) or len(os.listdir(training_selected_genes_dir)) == 0:
            print("Directory " + str(training_selected_genes_dir) + " doesn't exists or is empty")
            exit(1)
        if not os.listdir(test_selected_genes_dir) or len(os.listdir(test_selected_genes_dir)) == 0:
            print("Directory " + str(test_selected_genes_dir) + " doesn't exists or is empty")
            exit(1)

        X_train, y_train = methods.load_selected_genes(training_selected_genes_dir)
        X_test, y_test = methods.load_selected_genes(test_selected_genes_dir)

        #SMOTE
        print("\n[SMOTE]")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        methods.pca(X_train, y_train)
        methods.pca(X_test, y_test)

        sm = SMOTE(sampling_strategy=1.0, random_state=42, n_jobs=-1)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        print(Counter(y_train_sm))

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
        h = .02  # step size in the mesh
        x_min, x_max = X_train_sm[:, 0].min() - .5, X_train_sm[:, 0].max() + .5
        y_min, y_max = X_train_sm[:, 1].min() - .5, X_train_sm[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        # Put the result into a color plot
        clf.fit(X_train_sm[:, :2], y_train_sm)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        plt.scatter(X_train_sm[:, 0], X_train_sm[:, 1], c=y_train_sm, cmap=cm_bright, edgecolors='k')
        # Plot the testing points
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()

    elif args.classification_method == "nn":
        # TODO
        pass
    else:
        sys.stderr.write("Invalid value for <classification_method>")
        exit(1)


if __name__ == "__main__":
    main()
