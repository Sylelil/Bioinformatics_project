import statistics
import sys
import pandas as pd
from tqdm import tqdm
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


def remove_genes_with_median_0(data_frame):
    """
    Description: Removes the genes for which the median calculated
                 on all patients (healthy + sick) is = 0
    :param data_frame:
    :return data_frame:
    :return removed_genes:
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
    :param data_frame_0:
    :param data_frame_1:
    :return:
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
    :param data_frame_0:
    :param data_frame_1:
    :return:
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
    :param data_frame_0:
    :param data_frame_1:
    :return:
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
    :param data_frame_0:
    :param data_frame_1:
    :param alpha:
    :return m_reduced_genes:
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
    :param data_frame_0:
    :param data_frame_1:
    :param alpha:
    :return:
    """
    dict = {
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
            dict['genes'].append(gene)

        if not np.isnan(pvalue) and pvalue <= alpha / len(data_frame_0.columns):
            dict['genes_b'].append(gene)
            dict['p_values_b'].append(pvalue)
            dict['t_values_b'].append(tvalue)

        if not np.isnan(pvalue) and pvalue < alpha / (len(data_frame_0.columns) - i + 1):
            dict['genes_hh'].append(gene)

        if not np.isnan(pvalue) and pvalue < alpha / (len(data_frame_0.columns) - i):
            dict['genes_bh'].append(gene)

        dict['all_p_values'].append(pvalue)
        dict['all_t_values'].append(tvalue)
        i+=1

    return dict


def normalize_with_GeoMean_and_SizeFactor(data_frame):
    geo_mean = []
    removed_genes = []

    for gene in tqdm(data_frame.columns, desc=">> Compute geometric mean for each gene...", file=sys.stdout):
        gm = stats.mstats.gmean(data_frame[gene])
        if gm == 0:
            removed_genes.append(gene)
        else:
            geo_mean.append(gm)
    data_frame = data_frame.drop(columns=removed_genes)
    df_ratio = pd.DataFrame()
    df_ratio = df_ratio.append(data_frame)
    i = 0
    for gene in tqdm(df_ratio.columns, desc=">> Compute ratio for each gene...", file=sys.stdout):
        df_ratio[gene] = data_frame[gene] / geo_mean[i]
        i += 1

    size_factor = []
    df_ratio_transpose = pd.DataFrame.transpose(df_ratio)

    for gene in tqdm(df_ratio_transpose.columns, desc=">> Compute size factor for each samples...", file=sys.stdout):
        size_factor.append(statistics.median(df_ratio_transpose[gene]))
    data_frame_transpose = pd.DataFrame.transpose(data_frame)
    i = 0
    for gene in tqdm(data_frame_transpose.columns, desc=">> Compute final normalization for each gene...",
                     file=sys.stdout):
        data_frame_transpose[gene] = data_frame_transpose[gene] / size_factor[i]
        i += 1

    data_frame_after_scaling = pd.DataFrame.transpose(data_frame_transpose)

    return data_frame, data_frame_after_scaling, removed_genes


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

