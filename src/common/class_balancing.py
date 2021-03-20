from imblearn.over_sampling import RandomOverSampler

def random_upsampling(X,y):
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def smote_tomek(X,y):
    pass


def smote(X,y):
    pass


def prototype_generation(X,y):
    pass


def balance_dataset(X,y, method):
    if method == 'random_upsampling':
        X_balanced, y_balanced = random_upsampling(X,y)
    elif method == 'combined':
        X_balanced, y_balanced = smotetomek(X,y)
    elif method == 'smote':
        X_balanced, y_balanced = smote(X,y)
    elif method == 'downsampling':
        X_balanced, y_balanced = prototype_generation(X,y)
    return X_balanced, y_balanced
