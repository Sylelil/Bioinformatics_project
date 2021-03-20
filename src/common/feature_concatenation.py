import numpy as np


def concatenate(tile_features, gene_features, gene_copy_ratio=1):
    """
       Description: Concatenate the features of a tile with the gene features of the corresponding patient.
       :param tile_features: Tile features
       :param gene_features: Gene features
       :param gene_copy_ratio: Ratio by which gene features are copied to match the dimensionality of the tile. Default is 1.
       :return: Array of concatenated features
    """
    copied_gene_features = np.tile(gene_features, gene_copy_ratio)
    concatenated_features = np.concatenate((tile_features, copied_gene_features))
    return concatenated_features
