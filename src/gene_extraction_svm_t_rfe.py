import numpy
import pandas as pd
import statistics
import os
import sys
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from scipy import stats
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

def read_gene_expression_data(path):
    data_frame_0 = pd.DataFrame()
    data_frame_1 = pd.DataFrame()

    for file_name in tqdm(os.listdir(path), desc=">> Reading patient data...", file=sys.stdout):
        file_path = os.path.join(path, file_name)
        with open(file_path) as f:
            patient_df = pd.read_csv(f, sep="\t", header=None, index_col=0, names=[file_name.replace(".txt", "")])
            patient_df = pd.DataFrame.transpose(patient_df)
            if file_name.endswith("_0.txt"):
                data_frame_0 = data_frame_0.append(patient_df)
            else:
                data_frame_1 = data_frame_1.append(patient_df)
    return data_frame_0, data_frame_1


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
'''
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
    for gene in tqdm(data_frame_transpose.columns, desc=">> Compute final normalization for each gene...",file=sys.stdout):
        data_frame_transpose[gene] = data_frame_transpose[gene] / size_factor[i]
        i += 1

    data_frame_after_scaling = pd.DataFrame.transpose(data_frame_transpose)

    return data_frame, data_frame_after_scaling, removed_genes
'''

def welch_t_test(data_frame_0, data_frame_1, alpha):
    """
    Description: Parametric statistical test
    :param data_frame_0:
    :param data_frame_1:
    :param alpha:
    :return:
    """
    w_reduced_genes_bonferroni = []
    w_reduced_genes_holm_hochberg = []
    w_reduced_genes_benjamini_hochberg = []
    w_reduced_genes = []
    i = 0
    p_value = []
    t_values = []
    p_value_selected = []
    p_value_sorted = []
    sorted_gene_pvalue = []
    for gene in tqdm(data_frame_0.columns, desc=">> Computing test for each gene...", file=sys.stdout):
        tvalue, pvalue = stats.ttest_ind(np.array(data_frame_0[gene].tolist()),
                                         np.array(data_frame_1[gene].tolist()),
                                         equal_var=False, nan_policy='omit')
        if not np.isnan(pvalue) and pvalue < alpha:
           w_reduced_genes.append(gene)
        if not np.isnan(pvalue) and pvalue <= alpha / len(data_frame_0.columns):
            w_reduced_genes_bonferroni.append(gene)
            p_value_selected.append(pvalue)
        if not np.isnan(pvalue) and pvalue < alpha/(len(data_frame_0.columns)-i+1):
           w_reduced_genes_holm_hochberg.append(gene)
        if not np.isnan(pvalue) and pvalue < alpha/(len(data_frame_0.columns)-i):
            w_reduced_genes_benjamini_hochberg.append(gene)
        p_value.append(pvalue)
        t_values.append(tvalue)
    return w_reduced_genes, w_reduced_genes_bonferroni, w_reduced_genes_holm_hochberg, w_reduced_genes_benjamini_hochberg, p_value, p_value_selected, t_values

def main():
    #################################################
    #   1 - Reading data and exploratory analysis   #
    #################################################
    path_ge = sys.argv[1]

    # 1.a Reading data
    print("Reading gene expression data:")
    df_0, df_1 = read_gene_expression_data(path_ge)

    # 1.b Exploratory analysis
    print("\nExploratory analysis:")

    # 1.b.1 Compute number of samples
    n_samples_0 = len(df_0)
    n_samples_1 = len(df_1)
    print(f'>> Tumor samples: {n_samples_1}'
          f'\n>> Normal samples: {n_samples_0}')

    # 1.b.2 Compute number of features
    df = df_0.append(df_1, sort=False)  # Merge healthy data frame with diseased data frame
    n_features = len(df.columns)
    print(f">> Number of features (genes): {n_features}")
    print(df)

    #1.b.3 - Evaluate normality by skewness and kourt:

    n_skew_pos = 0
    n_skew_neg = 0
    n_kurt_1 = 0
    n_kurt_2 = 0
    for gene in tqdm(df.columns, desc=">> Evaluate asymmetry and kurt...", file=sys.stdout):
        if stats.skew(df[gene]) > 0.5:
            n_skew_pos += 1
        elif stats.skew(df[gene]) < -0.5:
            n_skew_neg += 1
        if stats.kurtosis(df[gene]) > 0:
            n_kurt_1 += 1
        elif stats.kurtosis(df[gene]) < 0:
            n_kurt_2 += 1

    print("La percentuale di geni con distribuzione asimmetrica (verso sx) è:", 100*(n_skew_pos/n_features))
    print("La percentuale di geni con distribuzione asimmetrica (verso dx) è:", 100*(n_skew_neg/n_features))
    print("La percentuale di geni con distribuzione platicurtica è:", 100*(n_kurt_2/n_features))
    print("La percentuale di geni con distribuzione leptocurtica è:", 100*(n_kurt_1/n_features))

    #Grafico a torta di skew and kurtosys


    ###################################################
    #   2 - Differentially gene expression analysis   #
    ###################################################

    # 2.a Pre-processing data
    print("\nDifferentially gene expression analysis [DGEA]")

    #2.a.1 - Pre-filtering: Remove genes with median = 0

    print("[DGEA pre-processing] Removing genes with median = 0:")
    df, removed_genes = remove_genes_with_median_0(df)
    n_features = len(df.columns)  # update number of features

    print(f'\n>> Number of genes removed: {len(removed_genes)}'
          f'\n>> Number of genes remained: {n_features}')

    #2.a.1 - Z-Normalization
    for gene in tqdm(df.columns, desc=">> Compute Z-normalization...", file=sys.stdout):
        df[gene] = stats.zscore(df[gene])
    df_0_transformed = df.loc[df.index.str.endswith('_0')]
    df_1_transformed = df.loc[df.index.str.endswith('_1')]


    '''
    #2.a.1 - Normalization with geometric mean and size factor : make no sense for FPKM data
    df_after_ratio, df_after_scaling, removed_genes = normalize_with_GeoMean_and_SizeFactor(df)
    n_features = len(df.columns)  # update number of features
    print(f'\n>> Number of genes removed: {len(removed_genes)}'
          f'\n>> Number of genes remained: {n_features}')
    print(f'>> First 10 values before transformation:\n{df.head(10)}')
    print(f'>> First 10 values after ratio geo_mean:\n{df_after_ratio.head(10)}')
    print(f'>> First 10 values after scaling:\n{df_after_scaling.head(10)}')
    df_0_after_scaling = df_after_scaling.loc[df_after_scaling.index.str.endswith('_0')]
    df_1_after_scaling = df_after_scaling.loc[df_after_scaling.index.str.endswith('_1')]

    # 2.a.2 - Normalization: Apply logarithmic transformation on gene expression data
    #                        Description : x = Log(x+1), where x is the gene expression value
    print(f'\n[DGEA pre-processing] Logarithmic transformation on gene expression data:'
          f'\n>> Computing logarithmic transformation...')
    df_log_transformed = df.applymap(lambda x: math.log(1+x, 2)) #base 10 o 2 è indifferente

    print(f'>> First 10 values before transformation:\n{df.head(10)}')
    print(f'>> First 10 values after transformation:\n{df_log_transformed.head(10)}')

    # Separate patients data frame in normal data frame and tumor data frame
    df_0_log_transformed = df_log_transformed.loc[df_log_transformed.index.str.endswith('_0')]
    df_1_log_transformed = df_log_transformed.loc[df_log_transformed.index.str.endswith('_1')]
    for gene in tqdm(df.columns, desc=">> Computing test for each gene...", file=sys.stdout):
        print(stats.describe(df[gene]))
    '''

    #2.b.2 Welch t test
    print("\n[DGEA statistical test] Welch t-test statistics:")
    alpha = 0.01
    w_reduced_genes, w_reduced_genes_b, w_reduced_genes_hh, w_reduced_genes_bh, w_pvalue, w_pvalue_selected, t_statistcs = welch_t_test(df_0_transformed, df_1_transformed, alpha)

    print(f'>> Number of selected genes with no correction (features): {len(w_reduced_genes)},'
          f'\n>> Number of selected genes with B (features): {len(w_reduced_genes_b)},'
          f'\n>> Number of selected genes with HH (features):{len(w_reduced_genes_hh)},'
          f'\n>> Number of selected genes with BH (features): {len(w_reduced_genes_bh)}')

    #Print p-value histogram: mi mostra qual'è la densità di frequenza dei valori di p-value sotto l'ipotesi nulla
    plt.hist(w_pvalue)
    plt.xlabel("p values Welch t-test")
    plt.show()
    # Print t-statistics per selezionare i top che hanno t-statistics più altro differente -> confronto con quelli del pvalue
    x = np.linspace(0, 30000, 28547)
    for i in range(len(t_statistcs)):
        t_statistcs[i] = abs(t_statistcs[i])
    sorted_t_statistics = []
    sorted_t_statistics = sorted(t_statistcs)
    plt.plot(x, sorted_t_statistics)
    plt.show()

    #selected gene with bigger statistics value (more differentialy expressed) -> it is the same of p_value selected?
    #are the perfromances better?
    selected_t_statistics = []
    for i in range(len(sorted_t_statistics)):
        if sorted_t_statistics[i] >= 15:
            selected_t_statistics.append(sorted_t_statistics[i])
    selected_genes = []
    t_statistcs_selected = []
    dict = {}
    for j, gene in enumerate(df.columns):
        dict[j] = gene
    for i in range(len(t_statistcs)):
        for j in range(len(selected_t_statistics)):
            if t_statistcs[i] == selected_t_statistics[j]:
                selected_genes.append(dict[i])
                t_statistcs_selected.append(t_statistcs[i])
    print(f'>> Number of most significant genes: {len(selected_genes)}')
    '''
    #Sort genes to respect p-value
    p_value_sorted = []
    sorted_gene = []
    p_value_sorted = sorted(w_pvalue_selected)
    for i in range(len(p_value_sorted)):
        for j in range(len(w_pvalue_selected)):
            if p_value_sorted[i] == w_pvalue_selected[j]:
                sorted_gene.append(w_reduced_genes_b[j])
    # For the chosen method:
    df_reduced = df[sorted_gene]
    n_features = len(df_reduced.columns)  # update number of features
    df_reduced_0 = df_reduced.loc[df_reduced.index.str.endswith('_0')]
    df_reduced_1 = df_reduced.loc[df_reduced.index.str.endswith('_1')]
    '''

    theta = 0.08
    df_reduced = df[selected_genes]
    genes_name = []
    for gene in df_reduced.columns:
        genes_name.append(gene)
    df_reduced_list = df_reduced.to_numpy()
    y = [x[-1:] for x in df_reduced.index]
    y = numpy.array(y)
    file_genes = open("genes4.txt", "w")
    ranked_genes = []
    ranked_t_statiscs = []
    #Rank
    while df_reduced_list.size > 0:
        model = SVC(kernel="linear")
        model.fit(df_reduced_list, y)
        weights = model.coef_
        ranking_scores = []
        for j in range(weights.shape[1]):
            r = theta * weights[:, j] + (1 - theta) * t_statistcs_selected[j]
            ranking_scores.append(r)
        threshold = max(ranking_scores)
        for k in range(len(ranking_scores)):
            if ranking_scores[k] == threshold:
                ranked_t_statiscs.append(t_statistcs_selected[k])
                t_statistcs_selected.remove(t_statistcs_selected[k])
                df_reduced_list = np.delete(df_reduced_list, k, 1)
                print(genes_name[k])
                ranked_genes.append(genes_name[k])
                file_genes.write("%s\n" % genes_name[k])
                genes_name.remove(genes_name[k])
    file_genes.close()

    reduced_ranked_genes = ranked_genes[0:200]
    t_statistics_top = ranked_t_statiscs[0:200]
    df_final = df_reduced[reduced_ranked_genes]
    df_final_list = df_final.to_numpy()
    CV = LeaveOneOut()
    y_true = []
    y_pred = []
    count_passi = 200
    accuracy = []
    weights = [[]]
    for i in tqdm(range(count_passi), desc="passi", file=sys.stdout):
        for train_ix, test_ix in CV.split(df_final_list):
            X_train, X_test = df_final_list[train_ix, :], df_final_list[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            model = SVC(kernel="linear")
            model.fit(X_train, y_train)
            weights = model.coef_
            pred = model.predict(X_test)
            y_true.append(y_test[0])
            y_pred.append(pred[0])
        global_acc = accuracy_score(y_true, y_pred)
        ranking_scores = []
        for j in range(weights.shape[1]):
            r = theta * weights[:, j] + (1 - theta) * t_statistics_top[j]
            ranking_scores.append(r)
        threshold = max(ranking_scores)
        for k in range(len(ranking_scores)):
            if ranking_scores[k] == threshold:
                t_statistics_top.remove(t_statistics_top[k])
                df_final_list = np.delete(df_final_list, k, 1)
        accuracy.append(global_acc)
        accuracy.reverse()
        y_true = []
        y_pred = []
        print(global_acc)
    x = np.linspace(0, 300, 200)
    plt.plot(x, accuracy)
    plt.show()

    '''
       for i in tqdm(range(count_passi), desc="passi", file=sys.stdout):
           for train_ix, test_ix in CV.split(df_reduced_list):
               X_train, X_test = df_reduced_list[train_ix, :], df_reduced_list[test_ix, :]
               y_train, y_test = y[train_ix], y[test_ix]
               model = SVC(kernel="linear")
               model.fit(X_train, y_train)
               weights = model.coef_
               pred = model.predict(X_test)
               y_true.append(y_test[0])
               y_pred.append(pred[0])
           global_acc = accuracy_score(y_true, y_pred)
           ranking_scores = []
           for j in range(weights.shape[1]):
               r = theta * weights[:, j] + (1 - theta) * t_statistcs_selected[j]
               ranking_scores.append(r)
           threshold = max(ranking_scores)
           for k in range(len(ranking_scores)):
               if ranking_scores[k] == threshold:
                   t_statistcs_selected.remove(t_statistcs_selected[k])
                   df_reduced_list = np.delete(df_reduced_list, k, 1)
                   print(genes_name[k])
                   file_genes.write("%s\n" % genes_name[k])
                   genes_name.remove(genes_name[k])

           accuracy.append(global_acc)
           y_true = []
           y_pred = []
           print(global_acc)
       '''

if __name__ == "__main__":
    main()