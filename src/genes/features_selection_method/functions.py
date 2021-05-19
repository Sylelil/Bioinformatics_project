import statistics
import sys
from tqdm import tqdm
from scipy import stats
import numpy as np


def remove_genes_with_median_0(data_frame):
    """
    Description: Removes the genes for which the median calculated
                 on all patients (normal + tumor) is = 0
    :param data_frame: data frame containing gene expression data
    :return data_frame: data frame after removing genes with median = 0
    :return removed_genes: removed genes names
    """
    removed_genes = []
    for gene in tqdm(data_frame.columns, desc=">> Compute median for each gene...", file=sys.stdout):
        median = statistics.median(data_frame[gene])
        if median == 0:
            removed_genes.append(gene)

    data_frame = data_frame.drop(columns=removed_genes)
    return data_frame, removed_genes


def mann_whitney_u_test(data_frame_0, data_frame_1, alpha):
    """
    Description: Non-parametric statistical test
    :param data_frame_0: data frame containing gene expression values of normal patients
    :param data_frame_1: data frame containing gene expression values of tumor patients
    :param alpha:
    :return m_reduced_genes: differentially expressed genes in normal and tumor samples
    """
    m_reduced_genes = []

    for gene in tqdm(data_frame_0.columns, desc=">> Computing test for each gene...", file=sys.stdout):
        statistic, pvalue = stats.mannwhitneyu(data_frame_0[gene].tolist(), data_frame_1[gene].tolist())
        if pvalue < alpha / len(data_frame_0.columns):  # Bonferroni adjustment
            m_reduced_genes.append(gene)

    return m_reduced_genes


def welch_t_test(data_frame_0, data_frame_1, alpha):
    """
    Description: Parametric statistical test
    :param data_frame_0: data frame containing gene expression values of normal patients
    :param data_frame_1: data frame containing gene expression values of tumor patients
    :param alpha:
    :return dict_: dictionary containing:
        - genes_b: differentially expressed genes in normal and tumor samples (with bonferroni adjustment)
        - genes: differentially expressed genes in normal and tumor samples
        - p_values_b: p values (bonferroni)
        - t_values_b: t values (bonferroni)
        - all_p_values: all p values
        - all_t_values: all t values
    """

    dict_ = {
        'genes_b': [],  # genes_bonferroni
        'genes': [],
        'p_values_b': [],  # pvalues bonferroni
        't_values_b': [],  # tvalues bonferroni
        'all_p_values': [],
        'all_t_values': []
    }

    for gene in tqdm(data_frame_0.columns, desc=">> Computing test for each gene...", file=sys.stdout):
        tvalue, pvalue = stats.ttest_ind(np.array(data_frame_0[gene].tolist()),
                                         np.array(data_frame_1[gene].tolist()),
                                         equal_var=False, nan_policy='omit')

        if not np.isnan(pvalue) and pvalue < alpha:
            dict_['genes'].append(gene)

        if not np.isnan(pvalue) and pvalue <= alpha / len(data_frame_0.columns):
            dict_['genes_b'].append(gene)
            dict_['p_values_b'].append(pvalue)
            dict_['t_values_b'].append(tvalue)

        dict_['all_p_values'].append(pvalue)
        dict_['all_t_values'].append(tvalue)

    return dict_


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

