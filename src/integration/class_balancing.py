from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import ClusterCentroids


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

