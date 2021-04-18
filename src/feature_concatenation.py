import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import utils

'''
def concatenate(tile_features, gene_features, gene_copy_ratio=1):
    """
       Description: Concatenate the features of a tile with the gene features of the corresponding patient.
       :param tile_features: Tile features
       :param gene_features: Gene features
       :param gene_copy_ratio: Ratio by which gene features are copied to match the dimensionality of the tile. Default is 1.
       :return: Array of concatenated features
    """

    if gene_copy_ratio == 1:
        df_genes = gene_features
    else:
        np_gene_features = gene_features.to_numpy()
        # separate caseids and labels from features
        caseids_labels = np_gene_features[:, -2:]
        features = np_gene_features[:, :-2]
        # copy features by ratio
        features_copied = np.tile(features, gene_copy_ratio)
        # re-append caseids and labels to copied features
        np_genes = np.append(features_copied, caseids_labels, axis=1)
        # convert back to dataframe
        col_names = [f'feat_g_{x}' for x in range(features_copied.shape[0])]
        col_names.extend(['caseid', 'label'])
        df_genes = pd.DataFrame(np_genes, columns=col_names).set_index('caseid')

    # perform (inner) join between dataframes based on caseids (indexes)
    #   since there are multiple tile feature lists for a single caseid, and only one gene feature list for a
    #   single caseid, that gene feature list will be duplicated and attached to each tile feature list with that caseid
    concatenated_features = tile_features.merge(df_genes, how='inner', on=['caseid', 'label'])

    data = concatenated_features[[x for x in concatenated_features.columns if x.startswith('feat')]]
    labels = concatenated_features['label'].values
    data_info = concatenated_features[[x for x in concatenated_features.columns if not x.startswith('feat') and x != 'label']]

    return data, labels, data_info
'''


def concatenate_data(lookup_dir_tiles, lookup_dir_genes, path_to_save, gene_copy_ratio=1):
    filepath_data = Path(path_to_save) / 'concat_data.csv'
    filepath_data_info = Path(path_to_save) / 'concat_data_info.csv'

    # get gene data
    print('>> Reading gene data...')
    df_gene_features = utils.get_gene_features(lookup_dir_genes)
    # shape of df_gene_features:
    #   [feat_g_1, feat_g_2, feat_g_3, ..., caseid, label]

    print('>> Checking for duplicates...')
    duplicates = df_gene_features[df_gene_features.index.duplicated()].index.values.tolist()
    if len(duplicates) >= 1:
        print(f"error: found {len(duplicates)} duplicated caseids: {duplicates}")
        print(f">> Ignoring duplicates...")
    df_gene_features = df_gene_features.drop(index=duplicates)

    # copy gene data according to ratio:
    print('>> Copy gene data according to ratio...')
    if gene_copy_ratio == 1:
        df_genes_copied = df_gene_features
    else:
        np_gene_features = df_gene_features.to_numpy()
        # separate caseids and labels from features
        caseids_labels = np_gene_features[:, -2:]
        features = np_gene_features[:, :-2]
        # copy features by ratio
        features_copied = np.tile(features, gene_copy_ratio)
        # re-append caseids and labels to copied features
        np_genes = np.append(features_copied, caseids_labels, axis=1)
        # convert back to dataframe
        col_names = [f'feat_g_{x}' for x in range(features_copied.shape[0])]
        col_names.extend(['caseid', 'label'])
        df_genes_copied = pd.DataFrame(np_genes, columns=col_names).set_index('caseid')

    # read tile features of each slide:
    print('>> Reading slide data and concatenating with gene data...')
    with open(filepath_data, mode='w') as f_data, open(filepath_data_info, mode='w') as f_data_info:
        i=0
        for np_file in tqdm(os.listdir(lookup_dir_tiles)):
            file_path = os.path.join(lookup_dir_tiles, np_file)
            filename = os.path.splitext(np_file)[0]
            caseid = filename[:-2]
            if caseid in duplicates:
                continue

            # get list of tile features of a single slide from file
            slide_data = np.load(file_path)
            # shape of slide_data:
            #   [[coordx_1,coordy_1, feat_t_1_1, feat_t_2_1, feat_t_3_1, ...]   # tile 1
            #    [coordx_2,coordy_2, feat_t_1_2, feat_t_2_2, feat_t_3_2, ...]   # tile 2
            #    [coordx_3,coordy_3, feat_t_1_3, feat_t_2_3, feat_t_3_3, ...]   # tile 3
            #    [...]]                                                         # tile ...

            # concatenation: append gene data with that caseid to each row of slides dataframe
            gene_data = df_genes_copied.loc[caseid, :].values.tolist()
            gene_data_list = [gene_data] * slide_data.shape[0]
            ''' 
            print(f'\ncaseid: {caseid}')
            print(f'gene_data: {gene_data}')
            print(f'slide_data shape: {slide_data.shape}')
            print(f'gene_data shape: {len(gene_data)}')
            print(f'gene_data_list shape: {np.asarray(gene_data_list).shape}')
            '''
            concatenated_data = np.append(slide_data, gene_data_list, axis=1)
            # shape of concatenated_data:
            #   [[coordx_1,coordy_1, feat_t_1_1, feat_t_2_1, ... , feat_g_1, feat_g_2, ..., label]   # tile 1
            #    [coordx_2,coordy_2, feat_t_1_2, feat_t_2_2, ... , feat_g_1, feat_g_2, ..., label]   # tile 2
            #    [coordx_3,coordy_3, feat_t_1_3, feat_t_2_3, ... , feat_g_1, feat_g_2, ..., label]   # tile 3
            #    [...]]                                                                              # tile ...

            data_info = concatenated_data[:, [0, 1, -1]]
            caseid_column = [caseid] * data_info.shape[0]
            caseid_column_np = np.asarray([caseid_column])
            data_info_2 = np.append(data_info, caseid_column_np.T, axis=1)

            concat_features_labels = concatenated_data[:, 2:]

            #save to csv
            if i==0:
                i=1
                # write column names at beginning of file
                f_data_info.write('coord1,coord2,label,caseid\n')
                col_names_data = [f'feat_{x}' for x in range(len(concat_features_labels[0]) - 1)]
                col_names_data.append('label\n')
                f_data.write(','.join(col_names_data))
            np.savetxt(f_data, concat_features_labels, delimiter=',', fmt='%s')
            np.savetxt(f_data_info, data_info_2, delimiter=',', fmt='%s')

    print(f'>> Concatenated features saved in in {filepath_data}')
    print(f'>> Data information saved in {filepath_data_info}')
    print('>> Done')


def main():
    tile_features_train_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'images' / 'train'
    tile_features_test_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'images' / 'test'
    tile_features_val_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'images' / 'val'
    gene_features_train_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'genes' / 'train'
    gene_features_test_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'genes' / 'test'
    gene_features_val_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'genes' / 'val'
    path_to_save = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results' / 'concatenated'
    path_to_save_train = Path(path_to_save) / 'train'
    path_to_save_test = Path(path_to_save) / 'test'
    path_to_save_val = Path(path_to_save) / 'val'

    if not os.path.exists(Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'results'):
        print("%s not existing." % Path('results'))
        exit()
    if not os.path.exists(path_to_save):
        print("%s not existing." % path_to_save)
        exit()
    if not os.path.exists(path_to_save_train):
        print("%s not existing." % path_to_save_train)
        exit()
    if not os.path.exists(path_to_save_test):
        print("%s not existing." % path_to_save_test)
        exit()
    if not os.path.exists(path_to_save_val):
        print("%s not existing." % path_to_save_val)
        exit()
    if not os.path.exists(tile_features_train_dir):
        print("%s not existing." % tile_features_train_dir)
        exit()
    if not os.path.exists(tile_features_test_dir):
        print("%s not existing." % tile_features_test_dir)
        exit()
    if not os.path.exists(tile_features_val_dir):
        print("%s not existing." % tile_features_val_dir)
        exit()
    if not os.path.exists(gene_features_train_dir):
        print("%s not existing." % gene_features_train_dir)
        exit()
    if not os.path.exists(gene_features_test_dir):
        print("%s not existing." % gene_features_test_dir)
        exit()
    if not os.path.exists(gene_features_val_dir):
        print("%s not existing." % gene_features_val_dir)
        exit()

    print('----------------------------')
    print(">> Concatenating train data:")
    print('----------------------------')
    concatenate_data(tile_features_train_dir, gene_features_train_dir, path_to_save_train)
    print('---------------------------------')
    print(">> Concatenating validation data:")
    print('---------------------------------')
    concatenate_data(tile_features_val_dir, gene_features_val_dir, path_to_save_val)
    print('---------------------------')
    print(">> Concatenating test data:")
    print('---------------------------')
    concatenate_data(tile_features_test_dir, gene_features_test_dir, path_to_save_test)




if __name__ == '__main__':
    main()