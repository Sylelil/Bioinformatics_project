import json
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.common import plots
import pandas as pd
from pathlib import Path
from sklearn import metrics
from config import paths

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def __compute_patch_score(y, y_pred):
    """
        Description: Private function. Compute the patch score, defined as the fraction of patches of the test set
                    that were correctly classified:
                    (num. of correctly classified patches)/(total num. of patches).
        :param y: list of ground truth labels.
        :param y_pred: list of predictions.
        :returns: patch_score
    """
    n_patches_correctly_classified = sum(pred == gt for gt, pred in zip(y, y_pred))
    tot_patches = len(y)
    patch_score = n_patches_correctly_classified / tot_patches
    return patch_score


def __compute_patient_score(y_pred_test, test_data_info_path):
    """
        Description: Private function. Compute the patient score, defined as the fraction of patches of a single patient
                    that were correctly classified (per-patient patch score), averaged over all the patients:
                    sum_i(patch score of the ith patient)/(total num. of patients).
        :param y_pred_test: list of predictions.
        :returns: patient_avg_score, patent_stddev_score
    """
    test_info_df = pd.read_csv(test_data_info_path)
    test_info_df['y_pred_test'] = y_pred_test
    test_filenames_file = Path(paths.filename_splits_dir) / 'test_filenames.npy'
    test_filenames = np.load(test_filenames_file)
    filename_list = []

    per_patient_patch_score_list = []
    for filename in test_filenames:
        if filename not in filename_list:
            filename_list.append(filename)
            patient_info = test_info_df.loc[test_info_df['filename'] == filename]
            y_pred_patient = list(patient_info['y_pred_test'])
            y_patient = list(patient_info['label'])
            per_patient_patch_score = __compute_patch_score(y_patient, y_pred_patient)
            per_patient_patch_score_list.append(per_patient_patch_score)
        else:  # filename of patient already aggregated
            continue

    patient_avg_score = np.mean(per_patient_patch_score_list)
    patent_stddev_score = np.std(per_patient_patch_score_list)

    return patient_avg_score, patent_stddev_score


def __classification_report(y_test, y_pred_test, test_scores, test_data_info_path=None, patch_classification=True):
    """
       Description: Private function. Generate classification report.
       :param y_test: ground truth test labels.
       :param y_pred_test: predicted test labels.
       :param test_scores: dictionary with scores of test classification.
       :returns: report, per_class_report
    """
    report = {}

    # compute confusion matrix values:
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    report['Normal Detected (True Negatives)'] = confusion_matrix[0][0]
    report['Normal Incorrectly Detected (False Positives)'] = confusion_matrix[0][1]
    report['Tumor Missed (False Negatives)'] = confusion_matrix[1][0]
    report['Tumor Detected (True Positives)'] = confusion_matrix[1][1]
    report['Total Tumor'] = np.sum(confusion_matrix[1])

    # add test scores:
    report.update(test_scores)

    # compute patch score and patient score:
    if patch_classification:
        report['patch_score'] = __compute_patch_score(y_test, y_pred_test)
        report['patient_avg_score'], report['patent_stddev_score'] = __compute_patient_score(y_pred_test, test_data_info_path)

    # compute further per-class scores:
    per_class_report = metrics.classification_report(y_test, y_pred_test)

    return report, per_class_report


def __print_classification_report(experiment_info, report, per_class_report, file=sys.stdout):
    """
       Description: Private function. Print classification report.
       :param experiment_info: dictionary with experiment information.
       :param file: file where report will be printed (default: sys.stdout).
       :param report: report.
       :param per_class_report: per class report.
    """
    print('Experiment details:', file=file)
    print('-------------------', file=file)
    for k, v in experiment_info.items():
        print("{:<50} {:<15}".format(k, v), file=file)
    print(file=file)

    print('Test results:', file=file)
    print('-------------', file=file)
    for k, v in report.items():
        print("{:<50} {:<15}".format(k, v), file=file)
    print(file=file)
    print(per_class_report, file=file)


def generate_classification_report(save_path, y_test, y_pred_test, test_scores, experiment_info, test_data_info_path=None, patch_classification=True):
    """
       Description: Generate classification report.
       :param test_data_info_path: test data info path.
       :param save_path: path to save report.
       :param y_test: ground truth test labels.
       :param y_pred_test: predicted test labels.
       :param test_scores: dictionary with scores of test classification.
       :param experiment_info: dictionary with experiment information.
       :param patch_classification: whether to compute patch score or not (default:true)
    """
    # generate report:
    print('>> Generating classification report...')
    report, per_class_report = __classification_report(y_test, y_pred_test, test_scores, test_data_info_path, patch_classification)

    # print on stdout:
    __print_classification_report(experiment_info, report, per_class_report)

    # print on file:
    report_path = Path(save_path) / 'report.txt'
    print(f'>> Saving classification report on file {report_path}...')
    with open(report_path, 'w') as f:
        __print_classification_report(experiment_info, report, per_class_report, file=f)


def generate_classification_plots(save_path, classifier, X_test, y_test, X_train, y_train):
    """
       Description: Generate classification plots: confusion matrix, ROC curve, Precision-Recall curve.
       :param X_train: train data.
       :param X_test: test data.
       :param classifier: classifier.
       :param save_path: path to save plots.
       :param y_test: ground truth test labels.
       :param y_train: ground truth train labels.
    """
    print(f'>> Generating classification plots and saving them in {save_path}...')

    #save data and classifier with pickle for final plots
    with open(Path(save_path) / 'X_test', 'wb') as xt_f, open(Path(save_path) / 'y_test', 'wb') as yt_f, open(
            Path(save_path) / 'classifier', 'wb') as clf_f:
        pickle.dump(X_test, xt_f)
        pickle.dump(y_test, yt_f)
        pickle.dump(classifier, clf_f)

    # plot confusion matrix
    plt.figure()
    metrics.plot_confusion_matrix(classifier, X_test, y_test)
    plt.savefig(Path(save_path) / 'confusion_matrix')

    # plot roc
    ax = plots.set_roc_prc_settings(title='ROC curve')
    metrics.plot_roc_curve(classifier, X_train, y_train, name='Train', ax=ax)
    metrics.plot_roc_curve(classifier, X_test, y_test, name='Test', ax=ax)
    plt.savefig(Path(save_path) / 'roc')

    # plot precision-recall curve
    ax = plots.set_roc_prc_settings(title='Precision-Recall curve')
    metrics.plot_precision_recall_curve(classifier, X_train, y_train, name='Train', ax=ax)
    metrics.plot_precision_recall_curve(classifier, X_test, y_test, name='Test', ax=ax)
    plt.savefig(Path(save_path) / 'prc')

    plt.show()


def generate_classification_plots_nn(save_path, y_test, y_pred_test, pred_proba_test, y_train, y_pred_train, pred_proba_train):
    """
       Description: Generate classification plots for NN: confusion matrix, ROC curve, Precision-Recall curve.
       :param pred_proba_train: probability of the sample for each class of train data.
       :param pred_proba_test: probability of the sample for each class of test data.
       :param save_path: path to save plots.
       :param y_test: ground truth test labels.
       :param y_pred_test: predicted test labels.
       :param y_train: ground truth train labels.
       :param y_pred_train: predicted train labels.
    """
    print(f'>> Generating classification plots and saving them in {save_path}...')

    # save data with pickle for final plots
    with open(Path(save_path) / 'pred_proba_test', 'wb') as ppt_f, open(Path(save_path) / 'y_test', 'wb') as yt_f, open(Path(save_path) / 'y_pred_test', 'wb') as ypt_f:
        pickle.dump(pred_proba_test, ppt_f)
        pickle.dump(y_test, yt_f)
        pickle.dump(y_pred_test, ypt_f)

    # plot confusion matrix
    plots.plot_cm(y_test, y_pred_test, save_path)
    plt.savefig(Path(save_path) / 'confusion_matrix')

    # plot roc
    ax = plots.set_roc_prc_settings(title='ROC curve')
    plots.plot_roc_nn("Train", y_train, pred_proba_train, ax)
    plots.plot_roc_nn("Test", y_test, pred_proba_test, ax)
    plt.savefig(Path(save_path) / 'roc')

    # plot precision-recall curve
    ax = plots.set_roc_prc_settings(title='Precision-Recall curve')
    plots.plot_prc_nn("Train", y_train, pred_proba_train, ax)
    plots.plot_prc_nn("Test", y_test, pred_proba_test, ax)
    plt.savefig(Path(save_path) / 'prc')

    plt.show()


def __get_experiment_title(dirname, clf_type):
    """
       Description: Get experiment title from name of folder.
       :param dirname: name of folder.
       :param clf_type: type of classifier
       :returns: title.
    """
    fields = dirname.split('_')
    title = ''
    if clf_type == 'nn':
        clf_fields = fields[2].split('+')
        title = f"MLP (n. hidden layers: 1, n. units: {clf_fields[3]}, lr:{clf_fields[1]}, epochs:{clf_fields[2]})"
        if clf_fields[0] == 'pcann':
            if fields[1] == 'None':
                title += "\ntrained on reduced image features + gene features"
            else:
                title += "\ntrained on reduced image features"
        else:
            if fields[1] == 'None':
                title += "\ntrained on all image features + gene features"
            else:
                title += "\ntrained on reduced image features"
    else:
        if fields[2] == 'sgdclassifier':
            title = 'SGDClassifier'
        elif fields[2] == 'linearsvc':
            title = 'LinearSVC'
    return title


def generate_final_classification_plots(results_path):
    """
       Description: Generate final classification plot, collecting results from folder. .
       :param results_path: folder containing experiment results.
    """
    # get data
    experiments = []
    for res_dir in tqdm(os.listdir(results_path)):
        path_res_dir = Path(results_path) / res_dir
        if os.path.isdir(path_res_dir):
            experiment = {}
            experiment['name'] = res_dir
            experiment['dir'] = path_res_dir
            if 'nn' in res_dir:
                experiment['type'] = 'nn'
                experiment['title'] = __get_experiment_title(os.path.basename(res_dir), 'nn')
            else:
                experiment['type'] = 'shallow'
                experiment['title'] = __get_experiment_title(os.path.basename(res_dir), 'shallow')
            experiments.append(experiment)

    # roc curves
    ax_roc = plots.set_roc_prc_settings(title='ROC curve')
    for experiment in experiments:
        if experiment['type'] == 'nn':
            with open(Path(experiment['dir']) / 'pred_proba_test', 'rb') as ppt_f, open(Path(experiment['dir']) / 'y_test', 'rb') as yt_f:
                pred_proba_test = pickle.load(ppt_f)
                y_test = pickle.load(yt_f)
                plots.plot_roc_nn(experiment['title'], y_test, pred_proba_test, ax_roc)
        else:
            with open(Path(experiment['dir']) / 'X_test', 'rb') as xt_f, open(
                    Path(experiment['dir']) / 'y_test', 'rb') as yt_f, open(
                    Path(experiment['dir']) / 'classifier', 'rb') as clf_f:
                X_test = pickle.load(xt_f)
                y_test = pickle.load(yt_f)
                classifier = pickle.load(clf_f)
                metrics.plot_roc_curve(classifier, X_test, y_test, name=experiment['title'], ax=ax_roc)
    plt.savefig(Path(results_path) / 'final_roc')

    # precision-recall curves
    ax_prc = plots.set_roc_prc_settings(title='Precision-Recall curve')
    for experiment in experiments:
        if experiment['type'] == 'nn':
            with open(Path(experiment['dir']) / 'pred_proba_test', 'rb') as ppt_f, open(
                    Path(experiment['dir']) / 'y_test', 'rb') as yt_f:
                pred_proba_test = pickle.load(ppt_f)
                y_test = pickle.load(yt_f)
                plots.plot_prc_nn(experiment['title'], y_test, pred_proba_test, ax_prc)
        else:
            with open(Path(experiment['dir']) / 'X_test', 'rb') as xt_f, open(
                    Path(experiment['dir']) / 'y_test', 'rb') as yt_f, open(
                    Path(experiment['dir']) / 'classifier', 'rb') as clf_f:
                X_test = pickle.load(xt_f)
                y_test = pickle.load(yt_f)
                classifier = pickle.load(clf_f)
                metrics.plot_precision_recall_curve(classifier, X_test, y_test, name=experiment['title'], ax=ax_prc)
    plt.savefig(Path(results_path) / 'final_prc')

    plt.show()

