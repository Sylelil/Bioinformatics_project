from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import metrics
from tensorflow import keras
from config import paths
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import ClusterCentroids

METRICS_skl = [
    metrics.accuracy_score,
    metrics.average_precision_score,
    metrics.f1_score,
    metrics.precision_score,
    metrics.recall_score,
    metrics.matthews_corrcoef,
    metrics.roc_auc_score,
    metrics.confusion_matrix,
]


METRICS_keras = [
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


def get_balancing_method(method, params):
    """
       Description: Return method corresponding to the parameter 'method'.
       :param method: Class balancing method.
       :param params: Parameters from configuration file.
       :returns: Class balancing method
    """
    if method == 'random_upsampling':
        return RandomOverSampler(random_state=params['general']['random_state'])
    elif method == 'combined':
        return SMOTEENN(random_state=params['general']['random_state'])
    elif method == 'smote':
        return SMOTE(random_state=params['general']['random_state'])
    elif method == 'downsampling':
        return ClusterCentroids(random_state=params['general']['random_state'])
    return None


def compute_class_weights(labels):
    """
        Description: Compute class weights from list of labels.
        :param labels: list of labels.
        :returns: class weights dictionary
    """
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


def compute_patch_score(y, y_pred):
    """
        Description: Compute the patch score, defined as the fraction of patches of the test set that were
                    correctly classified:
                    (num. of correctly classified patches)/(total num. of patches).
        :param y_pred_test: list of predictions.
    """
    n_patches_correctly_classified = sum(pred == gt for gt, pred in zip(y, y_pred))
    tot_patches = len(y)
    patch_score = n_patches_correctly_classified / tot_patches
    return patch_score


def compute_patient_score(y_pred_test):
    """
        Description: Compute the patient score, defined as the fraction of patches of a single patient that were
                    correctly classified (per-patient patch score), averaged over all the patients:
                    sum_i(patch score of the ith patient)/(total num. of patients).
        :param y_pred_test: list of predictions.

    """
    test_data_info_path = Path(paths.concatenated_results_dir) / 'test' / 'concat_data_info.csv'
    test_info_df = pd.read_csv(test_data_info_path)
    test_info_df['y_pred_test'] = y_pred_test
    # test_filenames_file = Path(paths.filename_splits_dir) / 'test_filenames.npy'
    test_filenames_file = Path(paths.filename_splits_dir) / 'test_caseids.npy'
    test_filenames = np.load(test_filenames_file)
    filename_list = []

    per_patient_patch_score_list = []
    for filename in tqdm(test_filenames):
        if filename not in filename_list:
            filename_list.append(filename)
            options = [f"{filename}_0", f"{filename}_1"]
            # patient_info = test_info_df.loc[test_info_df['filename'] == filename]
            patient_info = test_info_df.loc[test_info_df['filename'].isin(options)]
            y_pred_patient = list(patient_info['y_pred_test'])
            y_patient = list(patient_info['label'])
            per_patient_patch_score = compute_patch_score(y_patient, y_pred_patient)
            per_patient_patch_score_list.append(per_patient_patch_score)
        else:  # filename of patient already aggregated
            continue

    patient_avg_score = np.mean(per_patient_patch_score_list)
    patent_stddev_score = np.std(per_patient_patch_score_list)

    return patient_avg_score, patent_stddev_score


'''
    for filename in tqdm(test_filenames):
        if filename not in filename_list:
            filename_list.append(filename)
            patient_info = test_info_df.loc[test_info_df['filename'] == filename]
            patient_predictions = list(patient_info['y_pred_test'])
            gt_label = filename[-1]
            pred_label = 1 if 1 in patient_predictions else 0  # if at least one prediction is tumor, the predicted label is tumor
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)
        else:  # filename of patient already aggregated
            continue
            
    print('Test scores for results aggregated by patient:')
    test_scores_aggregated = []
    for metric in METRICS_skl:
        test_scores_aggregated.append((metric.__name__, metric(gt_labels, pred_labels)))
    for name, value in test_scores_aggregated:
        print(name, ': ', value)
    print()
    
    plots.plot_test_results_aggregated(gt_labels, np.asarray(pred_labels))
    plt.show()
'''


