import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_gene_features(lookup_dir):
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


def copy_gene_features(df_gene_features, gene_copy_ratio):
    """
       Description: Copy gene features of each patient according to specified ratio.
       :param df_gene_features: Dataframe of features
       :param gene_copy_ratio: copy ratio
       :returns: Dataframe of copied features.
    """
    print(f'>> Copying gene data according to ratio (={gene_copy_ratio})...')
    np_gene_features = df_gene_features.to_numpy()

    # separate filenames and labels from features
    filename_index = np.asarray(df_gene_features.index)
    labels_list = np_gene_features[:, -1:]
    labels_list = labels_list.reshape((len(labels_list),))
    filenames_labels = np.stack((filename_index, labels_list), axis=1)
    features = np_gene_features[:, :-1]

    # copy features by ratio
    features_copied = np.tile(features, gene_copy_ratio)

    # re-append filenames and labels to copied features
    np_genes = np.append(features_copied, filenames_labels, axis=1)

    # convert back to dataframe
    col_names = [f'feat_g_{x}' for x in range(features_copied.shape[1])]
    col_names.extend(['filename', 'label'])
    df_genes_copied = pd.DataFrame(np_genes, columns=col_names).set_index('filename')

    return df_genes_copied


def concatenate_copy_genes(lookup_dir_tiles, lookup_dir_genes, path_to_save, dataset_name, gene_copy_ratio):
    """
       Description: Concatenate gene features (possibly copied according to specified ratio) with tile features.
       :param lookup_dir_tiles: lookup directory with tile features
       :param lookup_dir_genes: lookup directory with gene features
       :param path_to_save: directory where concatenated data will be saved
       :param dataset_name: dataset name
       :param gene_copy_ratio: copy ratio for gene features
    """
    print()
    print('----------------------------------------------------------------------------------')
    print(f">> Concatenating {dataset_name} data (with gene copy ratio = {gene_copy_ratio}):")
    print('----------------------------------------------------------------------------------')

    filepath_data = Path(path_to_save) / f'x_{dataset_name}.csv'
    filepath_labels = Path(path_to_save) / f'y_{dataset_name}.csv'
    filepath_data_info = Path(path_to_save) / f'info_{dataset_name}.csv'
    num_files = 0
    num_positive = 0

    # get gene data
    print('>> Reading gene data...')
    df_gene_features = get_gene_features(lookup_dir_genes)
    # shape of df_gene_features:
    #   [feat_g_1, feat_g_2, feat_g_3, ..., filename, label]

    print('>> Checking for duplicates...')
    duplicates = df_gene_features[df_gene_features.index.duplicated()].index.values.tolist()
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated filenames: {duplicates}")
        exit()
        print(f">> Ignoring duplicates...")
        df_gene_features = df_gene_features.drop(index=duplicates)

    # copy gene data according to ratio:
    if gene_copy_ratio == 1:
        df_genes_copied = df_gene_features
    else:
        df_genes_copied = copy_gene_features(df_gene_features, gene_copy_ratio)

    print(f'>> Concatenating {dataset_name} slide data with gene data...')
    with open(filepath_data, mode='w') as f_data, \
            open(filepath_data_info, mode='w') as f_info, \
            open(filepath_labels, mode='w') as f_labels:
        i = 0
        for np_file in tqdm(os.listdir(lookup_dir_tiles)):
            num_files += 1
            file_path = os.path.join(lookup_dir_tiles, np_file)
            filename = os.path.splitext(np_file)[0]
            if filename.endswith('1'):
                num_positive += 1

            # get list of tile features of a single slide from file
            slide_data = np.load(file_path)
            # shape of slide_data:
            #   [[coordx_1,coordy_1, feat_t_1_1, feat_t_2_1, feat_t_3_1, ...]   # tile 1
            #    [coordx_2,coordy_2, feat_t_1_2, feat_t_2_2, feat_t_3_2, ...]   # tile 2
            #    [coordx_3,coordy_3, feat_t_1_3, feat_t_2_3, feat_t_3_3, ...]   # tile 3
            #    [...]]                                                         # tile ...
            slide_info = slide_data[:, [0, 1]]
            slide_features = slide_data[:, 2:]

            # concatenation: append gene data with that filename to each row of slides dataframe
            gene_data = df_genes_copied.loc[filename, :].values.tolist()
            gene_data_list = [gene_data] * slide_features.shape[0]

            concatenated_data = np.append(slide_features, gene_data_list, axis=1)
            # shape of concatenated_data:
            #   [[feat_t_1_1, feat_t_2_1, ... , feat_g_1, feat_g_2, ..., label]   # tile 1
            #    [feat_t_1_2, feat_t_2_2, ... , feat_g_1, feat_g_2, ..., label]   # tile 2
            #    [feat_t_1_3, feat_t_2_3, ... , feat_g_1, feat_g_2, ..., label]   # tile 3
            #    [...]]                                                           # tile ...
            concatenated_data_no_label = concatenated_data[:, :-1]
            labels_list = concatenated_data[:, -1:]
            # TODO FIT SCALER ON CONCATENATED FEATURES

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

    # TODO APPLY SCALER ON CONCATENATED FEATURES

    print()
    print(f'Total number of samples processed: {num_files}')
    print(f'Number of positive samples: {num_positive}')
    print(f'Number of negative samples: {num_files - num_positive}')
    print()
    print(f'Concatenated features saved in in {filepath_data}')
    print(f'Data information saved in {filepath_data_info}')
    print()
    print('>> Done.')


def concatenate_pca(lookup_dir_tiles, lookup_dir_genes, path_to_save, dataset_name, n_components=None, scaler=None, ipca=None, scaler_concatenated=None):
    """
       Description: Concatenate gene features (possibly copied according to specified ratio) with tile features.
       :param lookup_dir_tiles: lookup directory with tile features
       :param lookup_dir_genes: lookup directory with gene features
       :param path_to_save: directory where concatenated data will be saved
       :param dataset_name: dataset name
       :param n_components: number of Principal Components
       :param scaler: scaler object
       :param scaler_concatenated: scaler object for concatenated data
       :param ipca: incremental PCA object
    """
    print()
    print('--------------------------------------------')
    print(f">> Concatenating {dataset_name} data:")
    print('--------------------------------------------')

    filepath_data = Path(path_to_save) / f'x_{dataset_name}.csv'
    filepath_labels = Path(path_to_save) / f'y_{dataset_name}.csv'
    filepath_data_info = Path(path_to_save) / f'info_{dataset_name}.csv'
    num_files = 0
    num_positive = 0

    # get gene data
    print(f'>> Reading {dataset_name} gene data...')
    df_gene_features = get_gene_features(lookup_dir_genes)
    # shape of df_gene_features:
    #   [feat_g_1, feat_g_2, feat_g_3, ..., filename, label]

    print('>> Checking for duplicates...')
    duplicates = df_gene_features[df_gene_features.index.duplicated()].index.values.tolist()
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated filenames: {duplicates}")
        exit()
        print(f">> Ignoring duplicates...")
        df_gene_features = df_gene_features.drop(index=duplicates)

    if dataset_name == 'train':

        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=n_components)

        # fit scaler
        print('>> Fitting standard scaler on train slide data...')
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

        print('>> Fitting incremental PCA on train slide data...')
        # transform with scaler and fit pca
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

        print('>> Fitting scaler on concatenated train data...')
        # transform with scaler and ipca, concatenate and fit scaler on concatenated data
        scaler_concatenated = StandardScaler()
        for np_file in tqdm(os.listdir(lookup_dir_tiles)):
            file_path = os.path.join(lookup_dir_tiles, np_file)
            filename = os.path.splitext(np_file)[0]
            slide_data = np.load(file_path)
            slide_features = slide_data[:, 2:]
            # apply scaler
            slide_features_scaled = scaler.transform(slide_features)
            # apply ipca
            slide_features_ipca = ipca.transform(slide_features_scaled)
            # concatenation: append gene data with that filename to each row of slides dataframe
            gene_data = df_gene_features.loc[filename, :].values.tolist()
            gene_data_list = [gene_data] * slide_features_ipca.shape[0]
            concatenated_data = np.append(slide_features_ipca, gene_data_list, axis=1)
            # shape of concatenated_data:
            #   [[feat_t_1_1, feat_t_2_1, ... , feat_g_1, feat_g_2, ..., label]   # tile 1
            #    [feat_t_1_2, feat_t_2_2, ... , feat_g_1, feat_g_2, ..., label]   # tile 2
            #    [feat_t_1_3, feat_t_2_3, ... , feat_g_1, feat_g_2, ..., label]   # tile 3
            #    [...]]
            concatenated_data_no_label = concatenated_data[:, :-1]
            scaler_concatenated.partial_fit(concatenated_data_no_label)

    # transform with scaler, transform with pca, concatenate, transform with concatenation scaler and save on file:
    print(f'>> Transforming {dataset_name} slide data with standard scaler and incremental PCA, then concatenating with {dataset_name} gene data and scaling...')
    with open(filepath_data, mode='w') as f_data,\
            open(filepath_data_info, mode='w') as f_info,\
            open(filepath_labels, mode='w') as f_labels:
        i = 0
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
            # apply ipca
            slide_features_ipca = ipca.transform(slide_features_scaled)

            # concatenation: append gene data with that filename to each row of slides dataframe
            gene_data = df_gene_features.loc[filename, :].values.tolist()
            gene_data_list = [gene_data] * slide_features_ipca.shape[0]
            concatenated_data = np.append(slide_features_ipca, gene_data_list, axis=1)
            # shape of concatenated_data:
            #   [[feat_t_1_1, feat_t_2_1, ... , feat_g_1, feat_g_2, ..., label]   # tile 1
            #    [feat_t_1_2, feat_t_2_2, ... , feat_g_1, feat_g_2, ..., label]   # tile 2
            #    [feat_t_1_3, feat_t_2_3, ... , feat_g_1, feat_g_2, ..., label]   # tile 3
            #    [...]]

            concatenated_data_no_label = concatenated_data[:, :-1]
            labels_list = concatenated_data[:, -1:]
            scaled_concatenated_data_no_label = scaler_concatenated.transform(concatenated_data_no_label)

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
            np.savetxt(f_data, scaled_concatenated_data_no_label, delimiter=',', fmt='%s')
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

    return scaler, ipca, scaler_concatenated