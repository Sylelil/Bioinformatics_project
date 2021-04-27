import os
import numpy as np
import pandas as pd
from pathlib import Path
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


def concatenate(lookup_dir_tiles, lookup_dir_genes, path_to_save, gene_copy_ratio=1):
    """
       Description: Concatenate gene features (possibly copied according to specified ratio) with tile features.
       :param lookup_dir_tiles: lookup directory with tile features
       :param lookup_dir_genes: lookup directory with gene features
       :param gene_copy_ratio: copy ratio for gene features
       :param path_to_save: directory where concatenated data will be saved
    """
    if gene_copy_ratio == 1:
        filepath_data = Path(path_to_save) / 'concat_data.csv'
        filepath_data_info = Path(path_to_save) / 'concat_data_info.csv'
    else:
        filepath_data = Path(path_to_save) / 'concat_data_copied.csv'
        filepath_data_info = Path(path_to_save) / 'concat_data_info_copied.csv'

    # get gene data
    print('>> Reading gene data...')
    df_gene_features = get_gene_features(lookup_dir_genes)
    # shape of df_gene_features:
    #   [feat_g_1, feat_g_2, feat_g_3, ..., filename, label]

    print('>> Checking for duplicates...')
    duplicates = df_gene_features[df_gene_features.index.duplicated()].index.values.tolist()
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated filenames: {duplicates}")
        print(f">> Ignoring duplicates...")
    df_gene_features = df_gene_features.drop(index=duplicates)

    # copy gene data according to ratio:
    if gene_copy_ratio == 1:
        df_genes_copied = df_gene_features
    else:
        df_genes_copied = copy_gene_features(df_gene_features, gene_copy_ratio)

    # read tile features of each slide:
    print('>> Reading slide data and concatenating with gene data...')
    with open(filepath_data, mode='w') as f_data, open(filepath_data_info, mode='w') as f_data_info:
        i = 0
        for np_file in tqdm(os.listdir(lookup_dir_tiles)):
            file_path = os.path.join(lookup_dir_tiles, np_file)
            filename = os.path.splitext(np_file)[0]
            if filename in duplicates:
                continue

            # get list of tile features of a single slide from file
            slide_data = np.load(file_path)
            # shape of slide_data:
            #   [[coordx_1,coordy_1, feat_t_1_1, feat_t_2_1, feat_t_3_1, ...]   # tile 1
            #    [coordx_2,coordy_2, feat_t_1_2, feat_t_2_2, feat_t_3_2, ...]   # tile 2
            #    [coordx_3,coordy_3, feat_t_1_3, feat_t_2_3, feat_t_3_3, ...]   # tile 3
            #    [...]]                                                         # tile ...

            # concatenation: append gene data with that filename to each row of slides dataframe
            gene_data = df_genes_copied.loc[filename, :].values.tolist()
            gene_data_list = [gene_data] * slide_data.shape[0]

            concatenated_data = np.append(slide_data, gene_data_list, axis=1)
            # shape of concatenated_data:
            #   [[coordx_1,coordy_1, feat_t_1_1, feat_t_2_1, ... , feat_g_1, feat_g_2, ..., label]   # tile 1
            #    [coordx_2,coordy_2, feat_t_1_2, feat_t_2_2, ... , feat_g_1, feat_g_2, ..., label]   # tile 2
            #    [coordx_3,coordy_3, feat_t_1_3, feat_t_2_3, ... , feat_g_1, feat_g_2, ..., label]   # tile 3
            #    [...]]                                                                              # tile ...

            data_info = concatenated_data[:, [0, 1, -1]]
            filename_column = [filename] * data_info.shape[0]
            filename_column_np = np.asarray([filename_column])
            data_info_2 = np.append(data_info, filename_column_np.T, axis=1)

            concat_features_labels = concatenated_data[:, 2:]

            # save to csv
            if i == 0:
                i = 1
                # write column names at beginning of file
                f_data_info.write('coord1,coord2,label,filename\n')
                col_names_data = [f'feat_{x}' for x in range(len(concat_features_labels[0]) - 1)]
                col_names_data.append('label\n')
                f_data.write(','.join(col_names_data))
            np.savetxt(f_data, concat_features_labels, delimiter=',', fmt='%s')
            np.savetxt(f_data_info, data_info_2, delimiter=',', fmt='%s')

    print(f'>> Concatenated features saved in in {filepath_data}')
    print(f'>> Data information saved in {filepath_data_info}')
    print('>> Done')
