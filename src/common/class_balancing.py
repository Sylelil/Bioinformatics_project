from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import ClusterCentroids


def random_upsampling(X,y):
    """
       Description: Over-sample the minority class by picking samples at random with replacement.
       :param X: Data
       :param y: Labels
       :return: Balanced data with labels
    """
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res


def smote(X,y):
    """
       Description: Perform SMOTE - Synthetic Minority Over-sampling Technique.
       :param X: Data
       :param y: Labels
       :return: Balanced data with labels
    """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def smote_enn(X,y):
    """
       Description: Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.
       :param X: Data
       :param y: Labels
       :return: Balanced data with labels
    """
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_resample(X, y)
    return X_res, y_res


def prototype_generation(X,y):
    """
       Description: Undersample the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. 
       :param X: Data
       :param y: Labels
       :return: Balanced data with labels
    """
    cc = ClusterCentroids(random_state=42)
    X_res, y_res = cc.fit_resample(X, y)
    return X_res, y_res


def balance_dataset(X,y, method):
    """
       Description: Balance the dataset using the selected method. 
       :param X: Data
       :param y: Labels
       :param method: Class balancing method
       :return: Balanced data with labels
    """
    if method == 'random_upsampling':
        X_balanced, y_balanced = random_upsampling(X,y)
    elif method == 'combined':
        X_balanced, y_balanced = smote_enn(X,y)
    elif method == 'smote':
        X_balanced, y_balanced = smote(X,y)
    elif method == 'downsampling':
        X_balanced, y_balanced = prototype_generation(X,y)
    else:
        X_balanced, y_balanced = X, y
    return X_balanced, y_balanced
