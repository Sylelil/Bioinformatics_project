from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from src.common import class_balancing
from src.common.integration_classification_methods import classification_preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def shallow_classifier(args, params, train_filepath, val_filepath, test_filepath):
    X_train, y_train, X_val, y_val, X_test, y_test = classification_preprocessing.compute_scaling_pca(params, train_filepath, val_filepath, test_filepath)

    if args.balancing and args.balancing != 'weights':
            print(f">> Applying class balancing with {args.balancing}...")
            balancer = class_balancing.get_balancing_method(args.balancing, params)
            X_train, y_train = balancer.fit_resample(X_train, y_train)

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
        score = params['general']['scoring'](y_val, y_pred)
        print(f"    Validation {params['general']['scoring'].__name__}: {score}")
        if score > best_score:
            best_score = score
            best_hyperparam = hyperparam

    print(f"Best {params['general']['scoring'].__name__} ({'C' if args.method == 'svc' else 'alpha'}={best_hyperparam}): {best_score}")
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
    test_score = params['general']['scoring'](y_test, y_pred_test)
    print(f"Test {params['general']['scoring'].__name__} = {test_score}")

    print(classification_report(y_test, y_pred_test))
    print('>> Done')
