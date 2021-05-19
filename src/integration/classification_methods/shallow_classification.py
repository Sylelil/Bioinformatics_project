import os
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from config import paths
from src.common import classification_report_utils
from src.common.plots import plot_2D_svm_decision_boundary, plot_2D_svm_decision_boundary_integration
from src.integration import utils
from src.integration.classification_methods import common
from sklearn import metrics
from src.common.classification_metrics import METRICS_skl


def get_classifier(hyperparam, method_name, balancing, random_state, max_iter=1000):
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


def shallow_classifier(args, params, data_path, n_features_images):
    """
       Description: Train and test shallow classifier, then show results.
       :param n_features_images: number of features of images to be considered.
       :param args: arguments.
       :param params: configuration parameters.
       :param data_path: data path.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = utils.get_concatenated_data(data_path, n_features_images)

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
    experiment_descr += f"_{params['general']['n_features_images']}"
    experiment_descr += f"_{args.classification_method}"
    experiment_descr += f"_{args.balancing}"
    results_path = Path(paths.integration_classification_results_dir) / experiment_descr
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # generate classification report:
    experiment_info = {}
    experiment_info['Data folder'] = str(data_path)
    experiment_info['Selected features'] = f'{n_features_images} image features, no gene features' if n_features_images else 'All'
    experiment_info['Classification method'] = str(args.classification_method)
    experiment_info['Class balancing method'] = str(args.balancing)
    experiment_info['Best hyperparameter'] = f"{'C' if args.classification_method == 'linearsvc' else 'alpha'}={best_hyperparam}"
    experiment_info['Best validation score'] = f"{metric.__name__}={best_score}"
    classification_report_utils.generate_classification_report(results_path, y_test, y_pred_test, test_scores, experiment_info)

    # generate plots:
    classification_report_utils.generate_classification_plots(results_path, best_classifier, X_test, y_test, X_train, y_train)
    print('>> Done')
