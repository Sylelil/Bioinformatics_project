import os
from pathlib import Path

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
import numpy as np
from config import paths
from src.common import classification_report_utils
from src.common.plots import plot_2D_svm_decision_boundary, plot_2D_svm_decision_boundary_integration
from src.integration import utils
from src.integration.classification_methods import common
from sklearn import metrics
from src.common.classification_metrics import METRICS_skl, top_metric_skl
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.pipeline import Pipeline


def get_classifier_old(hyperparam, method_name, balancing, random_state, max_iter=1000):
    """
       Description: Get classifier method.
       :param max_iter: maximum number of iterations.
       :param hyperparam: hyperparameter to be set.
       :param method_name: method name.
       :param balancing: class balancing method.
       :param random_state: random state.
       :returns: classifier method.
    """
    if method_name == 'linearsvc':
        classifier = LinearSVC(C=hyperparam,
                               max_iter=max_iter,
                               class_weight=('balanced' if balancing == 'weights' else None),
                               random_state=random_state)
    else:
        classifier = SGDClassifier(alpha=hyperparam,
                                   max_iter=max_iter,  # np.ceil(10**6 / n_samples)
                                   class_weight=('balanced' if balancing == 'weights' else None),
                                   random_state=random_state)
    return classifier


def shallow_classifier_old(args, params, data_path):
    """
       Description: Train and test shallow classifier, then show results.
       :param n_features_images: number of features of images to be considered.
       :param args: arguments.
       :param params: configuration parameters.
       :param data_path: data path.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = utils.get_concatenated_data(data_path)

    if args.balancing and args.balancing != 'weights':
        print(f">> Applying class balancing with {args.balancing}...")
        balancer = common.get_balancing_method(args.balancing, params)
        X_train, y_train = balancer.fit_resample(X_train, y_train)

    metric = metrics.recall_score
    # metric = metrics.matthews_corrcoef

    if args.classification_method == 'linearsvc':
        print(">> Finding best hyperparameter C for LinearSVC...")
        grid = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]  # C
    else:
        print(">> Finding best hyperparameter alpha for SGDClassifier...")
        grid = [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06]  # alpha

    best_score = -1
    best_hyperparam = None
    for hyperparam in grid:
        print(f"{'C' if args.classification_method == 'linearsvc' else 'alpha'}={hyperparam}:")
        classifier = get_classifier(hyperparam=hyperparam,
                                    method_name=args.classification_method,
                                    balancing=args.balancing,
                                    random_state=params['general']['random_state'],
                                    max_iter=params[args.classification_method]['max_iter'])
        classifier.fit(X_train, y_train)
        #plot_2D_svm_decision_boundary_integration(data_path, classifier, X_train, y_train, X_val, y_val)
        y_pred = classifier.predict(X_val)
        score = metric(y_val, y_pred)
        print(f"    Validation {metric.__name__}: {score}; \n{metrics.precision_score(y_val, y_pred)}\n{metrics.confusion_matrix(y_val, y_pred)}\n")
        if score > best_score:
            best_score = score
            best_hyperparam = hyperparam

    # evaluate the performance of the model on test
    print()
    print(
        f"Best {metric.__name__} ({'C' if args.classification_method == 'linearsvc' else 'alpha'}={best_hyperparam}): {best_score}")
    print()
    print(f">> Training with best {'LinearSVC' if args.classification_method == 'linearsvc' else 'SGDClassifier'} model...")
    best_classifier = get_classifier(hyperparam=best_hyperparam,
                                     method_name=args.classification_method,
                                     balancing=args.balancing,
                                     random_state=params['general']['random_state'])

    best_classifier.fit(X_train, y_train)
    print(">> Testing...")
    print()
    y_pred_test = best_classifier.predict(X_test)
    y_pred_train = best_classifier.predict(X_train)

    test_scores = {}
    for metr in METRICS_skl:
        test_scores[metr.__name__] = metr(y_test, y_pred_test)

    # path to save results:
    experiment_descr = f"{os.path.split(data_path)[1]}"
    experiment_descr += f"_{params['general']['use_features_images_only']}"
    experiment_descr += f"_{args.classification_method}"
    experiment_descr += f"_{args.balancing}"
    results_path = Path(paths.integration_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # generate classification report:
    experiment_info = {}
    experiment_info['Data folder'] = str(data_path)
    experiment_info['Selected features'] = f'Image features, no gene features' if params['general']['use_features_images_only'] else 'All'
    experiment_info['PCA'] = f"n. components={params['general']['num_principal_components']}" if params['general']['num_principal_components'] else 'No'
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['Class balancing method'] = str(args.balancing)
    experiment_info['Best hyperparameter'] = f"{'C' if args.classification_method == 'linearsvc' else 'alpha'}={best_hyperparam}"
    experiment_info['Best validation score'] = f"{metric.__name__}={best_score}"
    test_data_info_path = data_path / 'info_test.csv'
    classification_report_utils.generate_classification_report(results_path, y_test, y_pred_test, test_scores, experiment_info,
                                                               test_data_info_path=test_data_info_path)

    # generate plots:
    classification_report_utils.generate_classification_plots(results_path, best_classifier, X_test, y_test, X_train, y_train)
    print('>> Done')


def get_classifier(method_name, params):
    """
       Description: Get classifier method.
       :param method_name: method name.
       :param params: parameters.
       :returns: classifier method, grid.
    """
    if method_name == 'linearsvc':
        classifier = ('linearsvc', LinearSVC(max_iter=params['linearsvc']['max_iter'], random_state=params['general']['random_state']))
        grid = [{'linearsvc__C' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}]  # C
    else:
        classifier = ('sgdclassifier', SGDClassifier(max_iter=params['sgdclassifier']['max_iter'], random_state=params['general']['random_state']))
        grid = [{'sgdclassifier__alpha': [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06]}]  # alpha
    return classifier, grid


def shallow_classifier(args, params, data_path):
    # classification pipeline with scaler, SMOTE and classifier:
    classifier, param_grid = get_classifier(args.classification_method, params)
    pipe = Pipeline([('standardscaler', StandardScaler()), ('smote', SMOTE(random_state=params['general']['random_state'])), classifier])

    # get data
    X_train, y_train, X_test, y_test = utils.get_data(data_path)


    test_outer_results = []
    train_outer_results = []
    best_hyperparams = []

    metric = top_metric_skl

    cv_outer = KFold(n_splits=params['cv']['n_outer_splits'], shuffle=True, random_state=params['general']['random_state'])

    # nested cross validation for unbiased error estimation:
    print(">> Nested cross validation for unbiased error estimation...")
    for train_ix, test_ix in tqdm(cv_outer.split(X_train, y_train)):

        # split data in k_outer folds (one is test, the rest is trainval) for outer loop
        X_train_cv, X_test_cv = X_train[train_ix, :], X_train[test_ix, :]
        y_train_cv, y_test_cv = y_train[train_ix], y_train[test_ix]

        # inner cross validation procedure for grid search of best hyperparameters:
        # trainval will be split in k_inner folds (one is val, the rest is train)
        # use train and val to find best model
        cv_inner = KFold(n_splits=params['cv']['n_inner_splits'], shuffle=True,
                         random_state=params['general']['random_state'])
        search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv_inner, n_jobs=-1,
                              scoring='recall', refit=True, verbose=2)

        # outer cross validation procedure to evaluate the performance of the best estimator:
        # fit the best model on the whole trainval
        search.fit(X_train_cv, y_train_cv)
        best_model = search.best_estimator_
        best_score = search.best_score_
        best_param = search.best_params_

        # evaluate the performance of the model on test
        y_test_pred = best_model.predict(X_test_cv)
        test_score = metric(y_test_cv, y_test_pred)
        y_train_pred = search.predict(X_train_cv)
        train_score = metric(y_train_cv, y_train_pred)

        test_outer_results.append(test_score)
        train_outer_results.append(train_score)
        best_hyperparams.append(search.best_params_)

        print(f"Inner cv: ")
        print(f"    Best {metric.__name__} = {best_score}, best hyperparam = {best_param}")
        print(f"Outer cv: ")
        print(f"    Val {metric.__name__} = {test_score}) - ", end='')
        print(f"Val precision = {metrics.precision_score(y_test_cv, y_test_pred)}) - ", end='')
        print(f"Val accuracy = {metrics.accuracy_score(y_test_cv, y_test_pred)} - ", end='')
        print(f"Val matthews_corrcoef = {metrics.matthews_corrcoef(y_test_cv, y_test_pred)}) - ", end='')
        print(f"Train {metric.__name__} = {train_score}) - ", end='')
        print(f"Train precision = {metrics.precision_score(y_train_cv, y_train_pred)}) - ", end='')
        print(f"Train matthews_corrcoef = {metrics.matthews_corrcoef(y_train_cv, y_train_pred)})")

    # calculate the mean score over all K outer folds, and report as the generalization error
    global_test_score = np.mean(test_outer_results)
    global_test_std = np.std(test_outer_results)
    global_train_score = np.mean(train_outer_results)
    global_train_std = np.std(train_outer_results)
    print()
    print(f"Global validation {metric.__name__} = {str(global_test_score)} ({str(global_test_std)})")
    print(f"Global training {metric.__name__} = {str(global_train_score)} ({str(global_train_std)})")
    print("List of best hyperparameters to check stability: ")
    print(best_hyperparams)
    print()

    # simple cross validation to find the best model:
    print('>> Simple cross validation to find the best model...')
    cv = KFold(n_splits=params['cv']['n_inner_splits'], shuffle=True, random_state=params['general']['random_state'])
    search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='recall',
                          refit=True, verbose=2)

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_hyperparam = search.best_params_
    best_score = search.best_score_
    y_pred = best_model.predict(X_test)
    final_test_score = metric(y_test, y_pred)
    print(f"Final test {metric.__name__} = {final_test_score}")

    test_scores = {}
    for metr in METRICS_skl:
        test_scores[metr.__name__] = metr(y_test, y_pred)

    # path to save results:
    experiment_descr = f"{os.path.split(data_path)[1]}"
    experiment_descr += f"_{params['general']['use_features_images_only']}"
    experiment_descr += f"_{args.classification_method}"
    experiment_descr += f"_smote"
    results_path = Path(paths.integration_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # generate classification report:
    experiment_info = {}
    experiment_info['Data folder'] = str(data_path)
    experiment_info['Selected features'] = f'Image features, no gene features' if params['general']['use_features_images_only'] else 'All'
    experiment_info['PCA'] = f"n. components={params['general']['num_principal_components']}" if params['general']['num_principal_components'] else 'No'
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['Class balancing method'] = 'SMOTE'
    experiment_info['Error estimation:'] = '---------------------------------------'
    experiment_info['Global validation score'] = f"{metric.__name__}={str(global_test_score)} ({str(global_test_std)})"
    experiment_info['Global train score'] = f"{metric.__name__}={str(global_train_score)} ({str(global_train_std)})"
    experiment_info['Test results:'] = '---------------------------------------'
    experiment_info['Best cv hyperparameter'] = f"{'C' if args.classification_method == 'linearsvc' else 'alpha'}={best_hyperparam}"
    experiment_info['Best cv score'] = f"{metric.__name__}={best_score}"
    experiment_info['Final test score'] = f"{metric.__name__}={final_test_score}"
    test_data_info_path = data_path / 'info_test.csv'
    classification_report_utils.generate_classification_report(results_path, y_test, y_pred, test_scores, experiment_info,
                                                               test_data_info_path=test_data_info_path)

    # generate plots:
    classification_report_utils.generate_classification_plots(results_path, best_model, X_test, y_test, X_train, y_train)
    print('>> Done')