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


def anderson_normality_test(data_frame_0, data_frame_1):
    """
    Description: Anderson test
    :param data_frame_0: data frame containing gene expression values of normal patients
    :param data_frame_1: data frame containing gene expression values of tumor patients
    :return: genes names for which normality check holds
    """
    normal_genes = []
    for gene in tqdm(data_frame_0.columns, desc="Checking normality for each gene...", file=sys.stdout):
        statistic_0, pvalue_0, alpha_0 = stats.anderson(data_frame_0[gene])
        pvalue_0_max = max(pvalue_0)

        statistic_1, pvalue_1, alpha_1 = stats.anderson(data_frame_1[gene])
        pvalue_1_max = max(pvalue_1)

        if statistic_0 < pvalue_0_max and statistic_1 < pvalue_1_max:
            normal_genes.append(gene)

    return normal_genes


def shapiro_normality_test(data_frame_0, data_frame_1, alpha):
    """
    Description: Shapiro test
    :param alpha:
    :param data_frame_0: data frame containing gene expression values of normal patients
    :param data_frame_1: data frame containing gene expression values of tumor patients
    :return: genes names for which normality check holds
    """
    normal_genes = []
    for gene in tqdm(data_frame_0.columns, desc="Checking normality for each gene...", file=sys.stdout):
        statistic_0, pvalue_0 = stats.shapiro(data_frame_0[gene])
        statistic_1, pvalue_1 = stats.shapiro(data_frame_1[gene])

        if pvalue_0 > alpha and pvalue_1 > alpha:
            normal_genes.append(gene)

    return normal_genes


def normal_test(data_frame_0, data_frame_1, alpha):
    """
    Description: Normal test
    :param alpha:
    :param data_frame_0: data frame containing gene expression values of normal patients
    :param data_frame_1: data frame containing gene expression values of tumor patients
    :return: genes names for which normality check holds
    """
    normal_genes = []
    for gene in tqdm(data_frame_0.columns, desc="Checking normality for each gene...", file=sys.stdout):
        statistic_0, pvalue_0 = stats.normaltest(data_frame_0[gene])
        statistic_1, pvalue_1 = stats.normaltest(data_frame_1[gene])

        if pvalue_0 > alpha and pvalue_1 > alpha:
            normal_genes.append(gene)

    return normal_genes


def mann_whitney_u_test(data_frame_0, data_frame_1, alpha):
    """
    Description: Non-parametric statistical test
    :param data_frame_0: data frame containing gene expression values of normal patients
    :param data_frame_1: data frame containing gene expression values of tumor patients
    :param alpha:
    :return m_reduced_genes: genes names for which the test holds
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
        - genes_b: genes for which the test holds (with bonferroni adjustment)
        - genes_hh: genes holm hochberg
        - genes_bh' genes_benjamini_hochberg
        - genes: genes for which the test holds
        - p_values_b: p values (bonferroni)
        - t_values_b: t values (bonferroni)
        - all_p_values:
        - all_t_values:
    """

    dict_ = {
        'genes_b': [],  # genes_bonferroni
        'genes_hh': [],  # genes holm hochberg
        'genes_bh': [],  # genes_benjamini_hochberg
        'genes': [],
        'p_values_b': [],  # pvalues bonferroni
        't_values_b': [],  # tvalues bonferroni
        'all_p_values': [],
        'all_t_values': []
    }

    i = 0

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

        if not np.isnan(pvalue) and pvalue < alpha / (len(data_frame_0.columns) - i + 1):
            dict_['genes_hh'].append(gene)

        if not np.isnan(pvalue) and pvalue < alpha / (len(data_frame_0.columns) - i):
            dict_['genes_bh'].append(gene)

        dict_['all_p_values'].append(pvalue)
        dict_['all_t_values'].append(tvalue)
        i+=1

    return dict_


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

