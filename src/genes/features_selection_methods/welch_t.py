import os
from . import common


def genes_selection_welch_t(df, params, results_dir):
    """
        Description: Welch's t-test to perform genes selection

        :param df: DataFrame, shape = [n_samples, n_features],
            where n_samples is the number of samples and n_features is the number of features.
            - DataFrame.columns: contains the gene names
            - DataFrame.index: contains the case_ids
        :param params: Dictionary
            configuration file parameters
        :param results_dir: Path
            directory to save results
        :return welch_dict['genes_b']: array-like, shape = [num_selected_features]
            differentially expressed genes in normal and tumor samples
    """

    # Pre-filtering: Remove genes with median = 0
    print("[DGEA pre-processing] Removing genes with median = 0:")
    df, removed_genes = common.remove_genes_with_median_0(df)
    n_features = len(df.columns)  # update number of features

    print(f'>> Number of genes removed: {len(removed_genes)}'
          f'\n>> Number of genes remained: {n_features}')

    df_0 = df.loc[df.index.str.endswith('0')]
    df_1 = df.loc[df.index.str.endswith('1')]

    # Welch t test
    print("\n[DGEA statistical test] Welch t-test statistics:")
    welch_dict = common.welch_t_test(df_0, df_1, params['alpha'])

    print(">> Number of selected genes with no correction (features) %d" % len(welch_dict['genes']))
    print(">> Number of selected genes with B (features) %d" % len(welch_dict['genes_b']))
    print(">> Number of selected genes with HH (features) %d" % len(welch_dict['genes_hh']))
    print(">> Number of selected genes with BH (features) %d" % len(welch_dict['genes_bh']))

    welch_t_bonferroni_genes_path = os.path.join(results_dir, "welch_t_bonferroni_genes.txt")
    fp = open(welch_t_bonferroni_genes_path, "w")
    for gene, t_value in zip(welch_dict['genes_b'], welch_dict['t_values_b']):
        fp.write("%s %f\n" % (gene, t_value))
    fp.close()

    return welch_dict['genes_b']
