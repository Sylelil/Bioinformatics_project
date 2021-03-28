import numpy as np
import pandas as pd


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
        col_names = [f'feat{x}' for x in range(features_copied.shape[0])]
        col_names.extend(['caseid', 'label'])
        df_genes = pd.DataFrame(np_genes, columns=col_names).set_index('caseid')

    # perform (inner) join between dataframes based on caseids (indexes)
    #   since there are multiple tile feature lists for a single caseid, and only one gene feature list for a
    #   single caseid, that gene feature list will be duplicated and attached to each tile feature list with that caseid
    concatenated_features = tile_features.merge(df_genes, left_index=True, right_index=True)
    labels = concatenated_features['label'].values

    return concatenated_features, labels
