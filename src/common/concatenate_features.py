import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def __get_gene_features(lookup_dir):
    """
       Description: Read extracted gene features from files.
       :param lookup_dir: Path of the lookup directory.
       :returns: Dataframe of features.
    """
    all_features = []  # list of all tile features
    data = None

    for np_file in tqdm(os.listdir(lookup_dir)):
        file_path = os.path.join(lookup_dir, np_file)
        filename = os.path.splitext(np_file)[0]
        label = filename[-1]

        # get features of a patient from file
        data = np.load(file_path)
        # shape of data:
        #   [feat1, feat2, feat3, ...]

        # append to each row of dataframe the filename and label of that slide
        data_filename_label = np.append(data, [filename, label])
        # shape of data_filename_label:
        #   [feat1, feat2, feat3, ..., filename, label]

        # add to list of all tile features
        data_list = list(data_filename_label)
        all_features.append(data_list)

    # convert to dataframe
    col_names = [f'feat_g_{x}' for x in range(data.shape[0])]
    col_names.extend(['filename', 'label'])
    df_gene_features = pd.DataFrame(all_features, columns=col_names).set_index('filename')

    print(f'shape: {df_gene_features.shape}')

    return df_gene_features


def save_features_images_only(path_to_save, lookup_dir_tiles_train=None, lookup_dir_tiles_val=None,
                              lookup_dir_tiles_test=None, dataset_name='train', n_components=None, scaler=None,
                              ipca=None, with_ipca=False):
    """
       Description: Save tile feature data in train and test files, applying scaler and possibly incremental PCA.
       :param path_to_save: Path to save.
       :param lookup_dir_tiles_train: lookup directory of train tiles.
       :param lookup_dir_tiles_val: lookup directory of validation tiles.
       :param lookup_dir_tiles_test: lookup directory of test tiles.
       :param dataset_name: dataset name.
       :param n_components: number of principal components.
       :param scaler: standard scaler method.
       :param ipca: incremental pca method.
       :param with_ipca: whether to use incremental PCA or not.
       :returns: scaler and incremental PCA objects.
    """
    print()
    print('--------------------------------------------')
    print(f">> Processing {dataset_name} data:")
    print('--------------------------------------------')

    filepath_data = Path(path_to_save) / f'x_{dataset_name}.csv'
    filepath_labels = Path(path_to_save) / f'y_{dataset_name}.csv'
    filepath_data_info = Path(path_to_save) / f'info_{dataset_name}.csv'
    num_files = 0
    num_positive = 0

    if dataset_name == 'train':
        scaler = StandardScaler()

        # fit scaler
        print('>> Fitting standard scaler on train and val slide data...')
        for lookup_dir_tiles in [lookup_dir_tiles_train, lookup_dir_tiles_val]:
            for np_file in tqdm(os.listdir(lookup_dir_tiles)):
                file_path = os.path.join(lookup_dir_tiles, np_file)
                # get list of tile features of a single slide from file
                slide_data = np.load(file_path)
                # shape of slide_data:
                #   [[coordx_1,coordy_1, feat_t_1_1, feat_t_2_1, feat_t_3_1, ...]   # tile 1
                #    [coordx_2,coordy_2, feat_t_1_2, feat_t_2_2, feat_t_3_2, ...]   # tile 2
                #    [coordx_3,coordy_3, feat_t_1_3, feat_t_2_3, feat_t_3_3, ...]   # tile 3
                #    [...]]
                slide_features = slide_data[:, 2:]
                scaler.partial_fit(slide_features)

        if with_ipca:
            ipca = IncrementalPCA(n_components=n_components)
            print('>> Fitting incremental PCA on train and val slide data...')
            # transform with scaler and fit pca
            for lookup_dir_tiles in [lookup_dir_tiles_train, lookup_dir_tiles_val]:
                slide_batch = []
                min_batch_dim = 2048
                for np_file in tqdm(os.listdir(lookup_dir_tiles)):
                    file_path = os.path.join(lookup_dir_tiles, np_file)
                    slide_data = np.load(file_path)
                    slide_features = slide_data[:, 2:]
                    slide_features_scaled = scaler.transform(slide_features)
                    slide_batch.extend(slide_features_scaled)
                    if len(slide_batch) >= min_batch_dim:
                        ipca.partial_fit(slide_batch)
                        slide_batch = []

    # transform with scaler, transform with pca and save on file:
    print(
        f'>> Transforming train and val slide data with standard scaler and incremental PCA, then saving as train dataset...')
    with open(filepath_data, mode='w') as f_data, \
            open(filepath_data_info, mode='w') as f_info, \
            open(filepath_labels, mode='w') as f_labels:
        i = 0
        if dataset_name == 'train':
            list_dir_tiles = [lookup_dir_tiles_train, lookup_dir_tiles_val]
        else:
            list_dir_tiles = [lookup_dir_tiles_test]

        for lookup_dir_tiles in list_dir_tiles:
            for np_file in tqdm(os.listdir(lookup_dir_tiles)):
                num_files += 1
                file_path = os.path.join(lookup_dir_tiles, np_file)
                filename = os.path.splitext(np_file)[0]
                if filename.endswith('1'):
                    num_positive += 1
                slide_data = np.load(file_path)
                slide_info = slide_data[:, [0, 1]]
                slide_features = slide_data[:, 2:]
                # apply scaler
                slide_features_scaled = scaler.transform(slide_features)
                if with_ipca:  # apply ipca
                    slide_features_scaled = ipca.transform(slide_features_scaled)

                # data info
                label_column = [filename[-1]] * slide_info.shape[0]
                label_column_np = np.asarray([label_column])
                filename_column = [filename] * slide_info.shape[0]
                filename_column_np = np.asarray([filename_column])
                data_info_2 = np.append(slide_info, filename_column_np.T, axis=1)
                data_info_2 = np.append(data_info_2, label_column_np.T, axis=1)
                # write header
                if i == 0:
                    i = 1
                    # write column names at beginning of file
                    f_info.write('coord1,coord2,filename,label\n')

                # save on file
                np.savetxt(f_info, data_info_2, delimiter=',', fmt='%s')
                np.savetxt(f_data, slide_features_scaled, delimiter=',', fmt='%s')
                np.savetxt(f_labels, label_column, delimiter=',', fmt='%s')

        print()
        print(f'Total number of samples processed: {num_files}')
        print(f'Number of positive samples: {num_positive}')
        print(f'Number of negative samples: {num_files - num_positive}')
        print()
        print(f'Concatenated features saved in in {filepath_data}')
        print(f'Data information saved in {filepath_data_info}')
        print()
        print('>> Done.')

        return scaler, ipca


def concatenate(path_to_save, dataset_name, lookup_dir_tiles_train=None, lookup_dir_tiles_val=None,
                lookup_dir_tiles_test=None, lookup_dir_genes_train=None, lookup_dir_genes_val=None,
                lookup_dir_genes_test=None, n_components=None, scaler=None, ipca=None, with_ipca=False):
    """
       Description: Concatenate gene features with tile features, possibly applying incremental PCA on tile features.
       Save in train and test files.
       :param path_to_save: directory where concatenated data will be saved.
       :param dataset_name: dataset name.
       :param lookup_dir_tiles_train: lookup directory with train tile features.
       :param lookup_dir_tiles_val: lookup directory with val tile features.
       :param lookup_dir_tiles_test: lookup directory with test tile features.
       :param lookup_dir_genes_train: lookup directory with train gene features.
       :param lookup_dir_genes_val: lookup directory with val gene features.
       :param lookup_dir_genes_test: lookup directory with test gene features.
       :param n_components: number of Principal Components.
       :param scaler: scaler object.
       :param ipca: incremental PCA object.
       :param with_ipca: whether to use incremental PCA or not.
       :returns: scaler for image features, incremental pca for image features.
    """
    print()
    print('--------------------------------------------')
    print(f">> Processing {dataset_name} data:")
    print('--------------------------------------------')

    filepath_data = Path(path_to_save) / f'x_{dataset_name}.csv'
    filepath_labels = Path(path_to_save) / f'y_{dataset_name}.csv'
    filepath_data_info = Path(path_to_save) / f'info_{dataset_name}.csv'
    num_files = 0
    num_positive = 0

    if dataset_name == 'train':

        scaler = StandardScaler()

        list_dir = [{'tiles': lookup_dir_tiles_train, 'genes': lookup_dir_genes_train},
                    {'tiles': lookup_dir_tiles_val, 'genes': lookup_dir_genes_val}]

        # fit scaler
        print('>> Fitting standard scaler...')
        for lookup_dir in list_dir:
            lookup_dir_tiles = lookup_dir['tiles']
            for np_file in tqdm(os.listdir(lookup_dir_tiles)):
                file_path = os.path.join(lookup_dir_tiles, np_file)
                # get list of tile features of a single slide from file
                slide_data = np.load(file_path)
                # shape of slide_data:
                #   [[coordx_1,coordy_1, feat_t_1_1, feat_t_2_1, feat_t_3_1, ...]   # tile 1
                #    [coordx_2,coordy_2, feat_t_1_2, feat_t_2_2, feat_t_3_2, ...]   # tile 2
                #    [coordx_3,coordy_3, feat_t_1_3, feat_t_2_3, feat_t_3_3, ...]   # tile 3
                #    [...]]
                slide_info = slide_data[:, [0, 1]]
                slide_features = slide_data[:, 2:]
                scaler.partial_fit(slide_features)

        if with_ipca:
            ipca = IncrementalPCA(n_components=n_components)
            print('>> Fitting incremental PCA...')
            # transform with scaler and fit pca
            for lookup_dir in list_dir:
                lookup_dir_tiles = lookup_dir['tiles']
                slide_batch = []
                min_batch_dim = 2048
                for np_file in tqdm(os.listdir(lookup_dir_tiles)):
                    file_path = os.path.join(lookup_dir_tiles, np_file)
                    slide_data = np.load(file_path)
                    slide_features = slide_data[:, 2:]
                    slide_features_scaled = scaler.transform(slide_features)
                    slide_batch.extend(slide_features_scaled)
                    if len(slide_batch) >= min_batch_dim:
                        ipca.partial_fit(slide_batch)
                        slide_batch = []

    else:  # dataset_name == 'test'
        list_dir = [{'tiles': lookup_dir_tiles_test, 'genes': lookup_dir_genes_test}]

    # transform with scaler, transform with pca, concatenate and save on file:
    print(
        f'>> Transforming {dataset_name} slide data with standard scaler and incremental PCA, then concatenating with {dataset_name} gene data...')
    with open(filepath_data, mode='w') as f_data, \
            open(filepath_data_info, mode='w') as f_info, \
            open(filepath_labels, mode='w') as f_labels:
        i = 0

        for lookup_dir in list_dir:
            lookup_dir_tiles = lookup_dir['tiles']
            lookup_dir_genes = lookup_dir['genes']

            # get gene data
            df_gene_features = __get_gene_features(lookup_dir_genes)
            # shape of df_gene_features:
            #   [feat_g_1, feat_g_2, feat_g_3, ..., filename, label]

            # get slide data
            for np_file in tqdm(os.listdir(lookup_dir_tiles)):
                num_files += 1
                file_path = os.path.join(lookup_dir_tiles, np_file)
                filename = os.path.splitext(np_file)[0]
                if filename.endswith('1'):
                    num_positive += 1
                slide_data = np.load(file_path)
                slide_info = slide_data[:, [0, 1]]
                slide_features = slide_data[:, 2:]
                # apply scaler
                slide_features_scaled = scaler.transform(slide_features)
                if with_ipca:  # apply ipca
                    slide_features_scaled = ipca.transform(slide_features_scaled)

                # concatenation: append gene data with that filename to each row of slides dataframe
                gene_data = df_gene_features.loc[filename, :].values.tolist()
                gene_data_list = [gene_data] * slide_features_scaled.shape[0]
                concatenated_data = np.append(slide_features_scaled, gene_data_list, axis=1)
                # shape of concatenated_data:
                #   [[feat_t_1_1, feat_t_2_1, ... , feat_g_1, feat_g_2, ..., label]   # tile 1
                #    [feat_t_1_2, feat_t_2_2, ... , feat_g_1, feat_g_2, ..., label]   # tile 2
                #    [feat_t_1_3, feat_t_2_3, ... , feat_g_1, feat_g_2, ..., label]   # tile 3
                #    [...]]

                concatenated_data_no_label = concatenated_data[:, :-1]
                labels_list = concatenated_data[:, -1:]

                # data info
                label_column = [filename[-1]] * slide_info.shape[0]
                label_column_np = np.asarray([label_column])
                filename_column = [filename] * slide_info.shape[0]
                filename_column_np = np.asarray([filename_column])
                data_info_2 = np.append(slide_info, filename_column_np.T, axis=1)
                data_info_2 = np.append(data_info_2, label_column_np.T, axis=1)
                # write header
                if i == 0:
                    i = 1
                    # write column names at beginning of file
                    f_info.write('coord1,coord2,filename,label\n')

                # save on file
                np.savetxt(f_info, data_info_2, delimiter=',', fmt='%s')
                np.savetxt(f_data, concatenated_data_no_label, delimiter=',', fmt='%s')
                np.savetxt(f_labels, labels_list, delimiter=',', fmt='%s')

    print()
    print(f'Total number of samples processed: {num_files}')
    print(f'Number of positive samples: {num_positive}')
    print(f'Number of negative samples: {num_files - num_positive}')
    print()
    print(f'Concatenated features saved in in {filepath_data}')
    print(f'Data information saved in {filepath_data_info}')
    print()
    print('>> Done.')

    return scaler, ipca
