import os
import sys
from collections import Counter
from pathlib import Path
import pandas as pd
from numpy.linalg import norm
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import matplotlib.pyplot as plt
from numpy import mean, std
from . import functions
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score


def genes_selection_svm_t_rfe(df, y, params, results_dir, config_dir):
    """
        Description: genes selection algorithm which extends support vector machine
            recursive feature elimination (SVM-RFE) algorithm by incorporating T-statistic

        :param df: DataFrame, shape = [n_samples, n_features],
            where n_samples is the number of samples and n_features is the number of features.
            - DataFrame.columns: contains the gene names
            - DataFrame.index: contains the case_ids
        :param y: array-like, shape = [n_samples]
            labels (0/1)
        :param params: Dictionary
            configuration file parameters
        :param results_dir: Path
            directory to save results
        :param config_dir: Path
            configuration file
        :return ranked_genes[:num_selected]: array-like, shape = [num_selected]
            genes names selected with SVM-T-RFE algorithm
    """

    # Pre-filtering: Remove genes with median = 0
    print("[DGEA pre-processing] Removing genes with median = 0:")
    df, removed_genes = functions.remove_genes_with_median_0(df)
    n_features = len(df.columns)  # update number of features

    print(f'>> Number of genes removed: {len(removed_genes)}'
          f'\n>> Number of genes remained: {n_features}')

    df_0 = df.loc[df.index.str.endswith('0')]
    df_1 = df.loc[df.index.str.endswith('1')]

    # Welch t test
    print("\n[DGEA statistical test] Welch t-test statistics:")
    welch_dict = functions.welch_t_test(df_0, df_1, params['alpha'])

    print(">> Number of selected genes with no correction (features) %d" % len(welch_dict['genes']))
    print(">> Number of selected genes with B (features) %d" % len(welch_dict['genes_b']))

    # Print t-statistics per selezionare i top che hanno t-statistics
    abs_t_statistics = [abs(x) for x in welch_dict['all_t_values']]
    sorted_t_statistics = sorted(abs_t_statistics)
    x = np.linspace(0, 30000, len(sorted_t_statistics))
    plt.plot(x, sorted_t_statistics)
    plt.savefig(Path(results_dir) / "sorted_t_stats.png")
    plt.show()
    plt.close()

    welch_t_bonferroni_genes_path = os.path.join(results_dir, "welch_t_bonferroni_genes.txt")
    fp = open(welch_t_bonferroni_genes_path, "w")
    for gene, t_value in zip(welch_dict['genes_b'], welch_dict['t_values_b']):
        fp.write("%s %f\n" % (gene, t_value))
    fp.close()

    # selected gene with bigger statistics value (more differentially expressed)
    zipped = zip(welch_dict['genes_b'], welch_dict['t_values_b'])  # creo lista di tuple (gene_name, t_value)
    selected_zipped = list(filter(lambda t: abs(t[1]) >= params['t_stat_threshold'],
                                  zipped))  # filtro la lista in base al valore di t_value in valore assoluto
    selected_genes, selected_t_statistics = map(list, zip(*selected_zipped))  # unzip and return two lists
    print(f'>> Number of most significant genes: {len(selected_genes)}')
    df_reduced = df[selected_genes]

    # svm t rfe
    print("\n[DGEA svm-t-rfe]:")

    # Rank
    c_values_file = str(params['alpha']) + "_" + str(params['t_stat_threshold']) + "_" + \
                    str(params['theta']) + "_" + str(params['cv_grid_search_rank']) + "_" + \
                    params['scoring_name'] + "_cvalues.txt"

    if params['smote']:
        scaler = StandardScaler()
        train_scaled_features = scaler.fit_transform(df_reduced.values)
        df_reduced = pd.DataFrame(train_scaled_features, index=df_reduced.index, columns=df_reduced.columns)
        print(df_reduced)

        sm = SMOTE(sampling_strategy=params['sampling_strategy'], random_state=params['random_state'], n_jobs=-1)
        df_reduced, y = sm.fit_resample(df_reduced, y)
        print(Counter(y))
        df_reduced["target"] = np.array([str(x) for x in y])
        print(list(df_reduced["target"]))
        df_reduced.index = list(df_reduced["target"])
        del df_reduced["target"]
        print(df_reduced)

    else:
        scaler = StandardScaler()
        train_scaled_features = scaler.fit_transform(df_reduced.values)
        df_reduced = pd.DataFrame(train_scaled_features, index=df_reduced.index, columns=df_reduced.columns)
        print(df_reduced)

    ranked_genes = ranking_genes(df_reduced, y, selected_t_statistics, params, Path(config_dir) / c_values_file)

    file_genes = open(Path(results_dir) / "ranked_genes.txt", "w")
    for gene_name in ranked_genes:
        file_genes.write("%s\n" % gene_name)
    file_genes.close()

    top_ranked_genes = ranked_genes[:params['top_ranked']]
    df_top_ranked_genes = df_reduced[top_ranked_genes]
    accuracy = accuracies_on_top_ranked_genes(df_top_ranked_genes, y, top_ranked_genes, params)

    x = np.arange(1, len(accuracy) + 1)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, len(x))
    plt.plot(x, accuracy)
    plt.xlabel("Number of top ranked genes")
    plt.ylabel(params['scoring_name'])
    plt.margins(x=0)
    plt.savefig(Path(results_dir) / "scores.png")
    plt.show()
    plt.close()

    num_selected = params['num_selected_genes']
    return ranked_genes[:num_selected]


def ranking_genes(df, y, selected_t_statistics, params, path_to_c_values):
    """
        Description: Feature ranking with recursive feature elimination.

        :param df: DataFrame, shape = [n_samples, n_selected_features],
            where n_samples is the number of samples and n_features is the number of selected features.
            - DataFrame.columns: contains the gene names
            - DataFrame.index: contains the case_ids
        :param y: array-like, shape = [n_samples]
            labels (0/1)
        :param selected_t_statistics: array-like, shape = [n_selected_features]
            t values from t test for the selected genes
        :param params: Dictionary
            configuration file parameters
        :param path_to_c_values: Path
            file containing C values obtained with GridSearchCV
        :return ranked_genes: array-like, shape = [n_selected_features]
            genes names ranked by feature importance
    """
    c_values = []
    perform_grid_search = False

    if os.path.exists(path_to_c_values):
        fp = open(path_to_c_values, 'r')
        lines = fp.readlines()
        for i, c in enumerate(lines):
            if functions.is_float(c):
                c_values.append(float(c))
            else:
                print(f'Line {i} is corrupt!')
                c_values.clear()
                break
        fp.close()

    if len(c_values) == 0:
        perform_grid_search = True

    df_array = df.to_numpy()
    genes_name = [gene for gene in df.columns]  # create list of gene names
    selected_t_statistics_norm = norm(selected_t_statistics)
    normalized_selected_t_statistics = [t / selected_t_statistics_norm for t in selected_t_statistics]
    ranked_genes = []
    ranked_norm_t_statistics = []
    param_grid = dict(svm__C=[0.0001, 0.001, 0.01, 0.1, 1])

    i = 0
    pbar = tqdm(desc=">> Ranking genes...", total=df_array.shape[1], file=sys.stdout)
    while df_array.shape[1] > 0:
        # ricerca del miglior C
        if perform_grid_search:
            pipe_grid = Pipeline([('svm', SVC(kernel='linear'))])
            cv = KFold(n_splits=params['cv_grid_search_rank'], shuffle=True, random_state=params['random_state'])
            grid = GridSearchCV(estimator=pipe_grid, param_grid=param_grid, cv=cv)
            grid.fit(df_array, y)  # Refit the estimator using the best found parameters on the whole dataset
            C = grid.best_params_['svm__C']
            c_values.append(C)
            best_model = grid.best_estimator_.named_steps['svm']
            weights = best_model.coef_[0]
        else:
            C = c_values[i]
            svm = SVC(kernel=params['kernel'], C=C)
            svm.fit(df_array, y)
            weights = svm.coef_[0]

        ranking_scores = []
        weights_norm = norm(weights)

        for j in range(len(weights)):
            r = params['theta'] * (abs(weights[j] / weights_norm)) + (1 - params['theta']) * (
                abs(normalized_selected_t_statistics[j]))
            ranking_scores.append(r)
        min_score = min(ranking_scores)  # sto prendendo il minimo

        indexes_to_remove = []
        for k in range(len(ranking_scores)):
            if ranking_scores[k] == min_score:  # rimuovo tutti i geni con score == min_score
                indexes_to_remove.append(k)

        for unwanted_index in sorted(indexes_to_remove, reverse=True):
            ranked_norm_t_statistics.append(normalized_selected_t_statistics[unwanted_index])
            del normalized_selected_t_statistics[unwanted_index]
            df_array = np.delete(df_array, unwanted_index, 1)
            ranked_genes.append(genes_name[unwanted_index])
            del genes_name[unwanted_index]
            pbar.update(1)
        i += 1
    pbar.close()
    ranked_genes.reverse()

    # salvo nel file i best C trovati
    if perform_grid_search:
        fp = open(path_to_c_values, "w")
        for c in c_values:
            fp.write("%f\n" % c)
        fp.close()

    return ranked_genes


def accuracies_on_top_ranked_genes(df_top_ranked_genes, y, top_ranked_genes, params):
    """
        Description: Determine the optimal number of highest-ranked features for classification accuracy

        :param df_top_ranked_genes: DataFrame, shape = [n_samples, n_top_ranked_features],
               where n_samples is the number of samples and n_features is the number of top ranked features.
               - DataFrame.columns: contains the gene names
               - DataFrame.index: contains the case_ids
        :param y: array-like, shape = [n_samples]
            labels (0/1)
        :param params: Dictionary
            configuration file parameters
        :param top_ranked_genes: array-like, shape = [n_top_ranked_features]
            top ranked genes names
        :return scores_list: array-like, shape = [n_top_ranked_features]
            mean scores considering from 1 to n_top_ranked_features
       """
    scores_list = []

    # costruisco la param_grid per la ricerca dei migliori hyperparms
    if params['kernel'] == 'rbf':
        # if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma
        C_range = [0.01, 0.1, 1]
        gamma_range = [0.001, 0.01, 0.1, 'scale']
        param_grid = dict(svm__gamma=gamma_range, svm__C=C_range)
    else:
        C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = dict(svm__C=C_range)

    # calcolo lo score considerando prima solo il primo top ranked gene, poi solo i primi due, etc..
    # fino ad arrivare a considerare tutti i top ranked genes
    for num_selected in tqdm(range(1, params['top_ranked'] + 1), file=sys.stdout):
        selected_features = top_ranked_genes[:num_selected]
        df_selected_array = df_top_ranked_genes[selected_features].to_numpy()

        # define search
        pipe_grid = Pipeline([('svm', SVC(kernel=params['kernel']))])
        cv_inner = KFold(n_splits=params['cv_grid_search_acc'], shuffle=True, random_state=params['random_state'])
        search = GridSearchCV(estimator=pipe_grid, param_grid=param_grid, cv=cv_inner)
        # configure the cross-validation procedure
        cv_outer = KFold(n_splits=params['cv_outer'], shuffle=True, random_state=params['random_state'])
        # execute the nested cross-validation
        scores = cross_val_score(search, df_selected_array, y, scoring=params['scoring'], cv=cv_outer)
        # report performance
        mean_score = mean(scores)
        scores_list.append(mean_score)
        print("\n" + params['scoring_name'] + " score " + str(mean_score))
        print(params['scoring_name'] + " std: " + str(std(scores)))

    return scores_list
