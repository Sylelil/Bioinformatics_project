import sys
import numpy as np
import matplotlib.pyplot as plt

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


def __compute_patient_score(y_pred_test, lookup_dir):
    """
        Description: Private function. Compute the patient score, defined as the fraction of patches of a single patient
                    that were correctly classified (per-patient patch score), averaged over all the patients:
                    sum_i(patch score of the ith patient)/(total num. of patients).
        :param y_pred_test: list of predictions.
        :returns: patient_avg_score, patent_stddev_score
    """
    if lookup_dir is None:
        test_data_info_path = Path(paths.concatenated_results_dir) / 'info_test.csv'
    else:
        test_data_info_path = Path(lookup_dir) / 'info_test.csv'
    test_info_df = pd.read_csv(test_data_info_path)
    test_info_df['y_pred_test'] = y_pred_test
    test_filenames_file = Path(paths.filename_splits_dir) / 'test_filenames.npy'
    test_filenames = np.load(test_filenames_file)
    filename_list = []

    per_patient_patch_score_list = []
    for filename in test_filenames:
        if filename not in filename_list:
            filename_list.append(filename)
            # options = [f"{filename}_0", f"{filename}_1"]
            # patient_info = test_info_df.loc[test_info_df['filename'].isin(options)]
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


def __classification_report(y_test, y_pred_test, test_scores, lookup_dir):
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
    report['patch_score'] = __compute_patch_score(y_test, y_pred_test)
    report['patient_avg_score'], patent_stddev_score = __compute_patient_score(y_pred_test, lookup_dir)

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


def generate_classification_report(save_path, y_test, y_pred_test, test_scores, experiment_info, lookup_dir=None):
    """
       Description: Generate classification report.
       :param save_path: path to save report.
       :param y_test: ground truth test labels.
       :param y_pred_test: predicted test labels.
       :param test_scores: dictionary with scores of test classification.
       :param experiment_info: dictionary with experiment information.
    """
    # generate report:
    print('>> Generating classification report...')
    report, per_class_report = __classification_report(y_test, y_pred_test, test_scores, lookup_dir)

    # print on stdout:
    __print_classification_report(experiment_info, report, per_class_report)

    # print on file:
    report_path = Path(save_path) / 'report.txt'
    print(f'>> Saving classification report on file {report_path}...')
    with open(report_path, 'w') as f:
        __print_classification_report(experiment_info, report, per_class_report, file=f)


def generate_classification_plots(save_path, y_test, y_pred_test, y_train, y_pred_train):
    """
       Description: Generate classification plots: confusion matrix, ROC curve, Precision-Recall curve.
       :param save_path: path to save plots.
       :param y_test: ground truth test labels.
       :param y_pred_test: predicted test labels.
       :param y_train: ground truth train labels.
       :param y_pred_train: predicted train labels.
    """
    print(f'>> Generating classification plots and saving them in {save_path}...')

    # plot confusion matrix
    plt.figure()
    plots.plot_cm(y_test, y_pred_test)
    plt.savefig(Path(save_path) / 'confusion_matrix')

    # plot roc
    plt.figure()
    plots.plot_roc("Train", y_train, y_pred_train)
    plots.plot_roc("Test", y_test, y_pred_test)
    plt.legend(loc='lower right')
    plt.savefig(Path(save_path) / 'roc')

    # plot precision-recall curve
    plt.figure()
    plots.plot_prc("Train", y_train, y_pred_train)
    plots.plot_prc("Test", y_test, y_pred_test)
    plt.legend(loc='lower right')
    plt.savefig(Path(save_path) / 'prc')

    plt.show()
