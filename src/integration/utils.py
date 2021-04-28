import configparser


def read_config_file(config_file_path):
    """
       Description: Read configuration parameters.
       :param config_file_path: Path of the configuration file.
       :returns: Dictionary of parameters
    """
    params = {}
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_file_path)

    # general
    params['general'] = {}
    random_state = config['general']['random_state']
    if random_state == 'None' or random_state == '':
        params['general']['random_state'] = None
    else:
        params['general']['random_state'] = config.getint('general', 'random_state')

    # pca
    params['pca'] = {}
    params['pca']['percentage_of_variance'] = config.getfloat('pca', 'percentage_of_variance')
    params['pca']['n_components'] = config.getint('pca', 'n_components')

    # preprocessing
    params['preprocessing'] = {}
    params['preprocessing']['batchsize'] = config.getint('preprocessing', 'batchsize')

    # nn
    params['nn'] = {}
    params['nn']['epochs'] = config.getint('nn', 'epochs')
    params['nn']['batchsize'] = config.getint('nn', 'batchsize')

    return params
