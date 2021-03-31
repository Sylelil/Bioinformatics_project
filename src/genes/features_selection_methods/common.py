import os
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


def plot_histograms(df_0, df_1, colors):
    """
    Function: Per-class feature histograms, for the genes discarded by statistical test
    Description: Create histograms, one for each feature, each one counting how many patients appear
                      with a feature (gene expression value) in a certain range (called bin).
                      Each plot contains 2 histograms one for the healthy patients and
                      the other for the diseased patients.
    :param colors:
    :param df_1:
    :param df_0:
    :return:
    """
    df = df_0.append(df_1)
    fig, axes = plt.subplots(15, 2, figsize=(10, 20))
    ax = axes.ravel()
    for i, gene in enumerate(df.columns):
        _, bins = np.histogram(df[gene], bins=50)
        ax[i].hist(df_1[gene], bins=bins, color=colors[0], alpha=.5)
        ax[i].hist(df_0[gene], bins=bins, color=colors[1], alpha=.9)
        ax[i].set_title(gene)
        ax[i].set_yticks(())
        if i == 29:
            break
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["malignant", "benign"], loc="best")
    fig.tight_layout()
    plt.show()


def plot_variance_vs_num_components(x):
    """
    Description : The curve quantifies how much of the total, 13595-dimensional variance
                  is contained within the first N components
    :param x:
    :return:
    """
    pca = PCA()
    pca.fit(x)
    cumsum = np.cumsum(pca.explained_variance_ratio_)  # Cumulative proportion of variance (from first PC to last PC)
    d = np.argmax(cumsum >= 0.95) + 1  # num of components required to preserve 95% of training set variance
    plt.plot(cumsum)
    plt.vlines(d, 0, cumsum[d], linestyles='dashed', linewidths=0.5)
    plt.hlines(cumsum[d], 0, d, linestyles='dashed', linewidths=0.5)
    plt.scatter(d, cumsum[d], facecolors='black', alpha=.9, s=25)
    plt.annotate('Elbow', xytext=(130, 0.65), xy=(100, 0.8), arrowprops={'facecolor': 'black'})
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.show()


def plot_hierarchical_clustering(df):
    """
    Description: Plot hierarchical clustering dendrogram and heatmap.
    :param df:
    :return:
    """
    # heatmap_data = pd.pivot_table(df, values=df.index, index=df.index, columns=df.columns)
    hcplot = sns.clustermap(df)
    plt.show()


def random_undersample(df, size):
    random_indices = np.random.choice(df.index, size, replace=False)
    df_undersampled = df.loc[random_indices]
    return random_indices, df_undersampled


def scree_plot(pca):
    plt.plot(pca.explained_variance_)
    plt.xlabel('Principal component')
    plt.ylabel('Eigenvalue')
    plt.show()


def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()


def plot_pca_2D(X_train_pca, y_train):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = ['0', '1']
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indices_to_keep = np.array(y_train == target)
        ax.scatter(X_train_pca[indices_to_keep, 1]  # first principal component
                   , X_train_pca[indices_to_keep, 2]  # second principal component
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

