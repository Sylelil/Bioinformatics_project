from pathlib import Path

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm

from config import paths
from src.common import class_balancing
from src.common.integration_classification_methods import classification_preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data_manipulation import concatenate_features

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

METRICS = [
    metrics.accuracy_score,
    metrics.average_precision_score,
    metrics.f1_score,
    metrics.precision_score,
    metrics.recall_score,
    metrics.matthews_corrcoef,
    metrics.roc_auc_score,
    metrics.confusion_matrix,
]

def shallow_classifier(args, params, train_filepath, val_filepath, test_filepath):
    """
       Description: Train and test shallow classifier, then show results.
       :param args: arguments.
       :param params: configuration parameters.
       :param train_filepath: train data path.
       :param val_filepath: validation data path.
       :param test_filepath: test data path.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = classification_preprocessing.compute_scaling_pca(params, train_filepath, val_filepath, test_filepath)

    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = class_balancing.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)
    if args.balancing:
        metric = metrics.accuracy_score
    else:
        metric = metrics.matthews_corrcoef

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
                                   random_state=params['general']['random_state'])
        else:
            classifier = SGDClassifier(alpha=hyperparam,
                                       max_iter=10, # np.ceil(10**6 / n_samples)
                                       class_weight=('balanced' if args.balancing == 'weights' else None),
                                       random_state=params['general']['random_state'])
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        score = metric(y_val, y_pred)
        print(f"    Validation {metric.__name__}: {score}")
        if score > best_score:
            best_score = score
            best_hyperparam = hyperparam

    print(f"Best {metric.__name__} ({'C' if args.method == 'svc' else 'alpha'}={best_hyperparam}): {best_score}")
    print(f">> Training with best {'LinearSVC' if args.method == 'svc' else 'SGDClassifier'} model...")
    if args.method == 'svc':
        best_classifier = LinearSVC(C=best_hyperparam,
                                    class_weight=('balanced' if args.balancing == 'weights' else None),
                                    random_state=params['general']['random_state'])
    else:
        best_classifier = SGDClassifier(alpha=best_hyperparam,
                                        max_iter=10,
                                        class_weight=('balanced' if args.balancing == 'weights' else None),
                                        random_state=params['general']['random_state'])
    best_classifier.fit(X_train, y_train)
    print(">> Testing...")
    y_pred_test = best_classifier.predict(X_test)

    print('Test scores:')
    test_scores = []
    for metric in METRICS:
        test_scores.append((metric.__name__, metric(y_test, y_pred_test)))
    for name, value in test_scores:
        print(name, ': ', value)
    print()

    print(metrics.classification_report(y_test, y_pred_test))

    # aggregate results for each patient in test dataset:
    print('>> Aggregating results for each patient in test dataset...')
    # test_data_info_path = Path(paths.concatenated_results_dir) / 'test' / 'concat_data_info.csv'
    test_data_info_path = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'concatenated_results' / 'test' / 'concat_data_info.csv'
    test_info_df = pd.read_csv(test_data_info_path)
    test_info_df['y_pred_test'] = y_pred_test
    # test_filenames_file = Path(paths.filename_splits_dir) / 'test_filenames.npy'
    test_filenames_file = Path('.') / 'assets' / 'caseid_splits' / 'test_caseids.npy'
    test_filenames = np.load(test_filenames_file)
    gt_labels = []
    pred_labels = []
    filename_list = []
    for filename in tqdm(test_filenames):
        if filename not in filename_list:
            options = [f"{filename}_0", f"{filename}_1"]
            # patient_info = test_info_df.loc[test_info_df['filename'] == filename]
            patient_info = test_info_df.loc[test_info_df['filename'].isin(options)]
            patient_predictions = list(patient_info['y_pred_test'])
            gt_label = list(patient_info['label'])[0]
            #gt_label = filename[-1]
            pred_label = 1 if 1 in patient_predictions else 0  # if at least one prediction is tumor, the predicted label is tumor
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)
        else:  # filename of patient already aggregated
            continue

    print('Test scores for results aggregated by patient:')
    test_scores_aggregated = []
    for metric in METRICS:
        test_scores_aggregated.append((metric.__name__, metric(gt_labels, pred_labels)))
    for name, value in test_scores_aggregated:
        print(name, ': ', value)
    print()

    print('>> Done')
