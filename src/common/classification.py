from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid

def get_svm(params):
    pca = PCA(random_state=params['random_state'], n_components=params['percentage_of_variance'])
    linearsvc = LinearSVC(random_state=params['random_state'])
    estimators = [pca, linearsvc]
    param_grid = {
        'linearsvc__C': [0.0001, 0.001, 0.01, 0.1, 1],
    }
    return estimators, ParameterGrid(param_grid)


def get_classifier_param_grid(method, params):
    if method == 'svm':
        return get_svm(params)
    if method == 'sgd':
        pass
    elif method == 'nn':
        pass  # TODO classify data with NN
    elif method == 'pca_nn':
        pass  # TODO classify data with PCA+NN



