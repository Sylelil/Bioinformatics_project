import math
import os
import numpy
import sys
import pandas as pd
import seaborn as sns
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
from scipy import stats
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold, StratifiedKFold
import matplotlib.pyplot as plt
from numpy import mean

from src.genes.features_selection_methods import common


def genes_selection_svm_t_rfe(df, params, results_dir, config_dir):
    # Pre-filtering: Remove genes with median = 0
    print("[DGEA pre-processing] Removing genes with median = 0:")
    df, removed_genes = common.remove_genes_with_median_0(df)
    n_features = len(df.columns)  # update number of features

    print(f'>> Number of genes removed: {len(removed_genes)}'
          f'\n>> Number of genes remained: {n_features}')

    # in questo modo applichiamo la z normalization ai gene expression data.
    # Tutte le operazioni successive quindi sono fatte sui dati normalizzati
    # in particolare:
    # - il welch t test è fatto sui dati normalizzati
    # - il ranking è fatto sui dati normalizzati
    # Ho provato a fare il ranking tenendo i dati non normalizzati ma il loop ad un certo punto si blocca (non riesce a fare il fit!!)
    # credo che per l'svm ci voglia una normalizzazione dei dati. Da verificare però.
    # come mai abbiamo scelto la z-normalization?
    # Z-Normalization
    '''
    df_trasformed = df.copy()
    print("\n[DGEA pre-processing] Z-normalization:")
    for gene in tqdm(df.columns, desc=">> Compute Z-normalization for each gene...", file=sys.stdout):
        df_trasformed[gene] = stats.zscore(df[gene])
    
    df_trasformed_0 = df_trasformed.loc[df_trasformed.index.str.endswith('_0')]
    df_trasformed_1 = df_trasformed.loc[df_trasformed.index.str.endswith('_1')]


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
    df_0 = df.loc[df.index.str.endswith('_0')]
    df_1 = df.loc[df.index.str.endswith('_1')]


    # Welch t test
    print("\n[DGEA statistical test] Welch t-test statistics:")
    welch_dict = common.welch_t_test(df_0, df_1, params['alpha'])

    print(">> Number of selected genes with no correction (features) %d" % len(welch_dict['genes']))
    print(">> Number of selected genes with B (features) %d" % len(welch_dict['genes_b']))
    print(">> Number of selected genes with HH (features) %d" % len(welch_dict['genes_hh']))
    print(">> Number of selected genes with BH (features) %d" % len(welch_dict['genes_bh']))

    # Print p-value histogram: mi mostra qual'è la densità di
    # frequenza dei valori di p-value sotto l'ipotesi nulla
    plt.hist(welch_dict['all_p_values'])
    plt.xlabel("p values Welch t-test")
    plt.show()

    # Print t-statistics per selezionare i top che hanno t-statistics
    # più altro differente -> confronto con quelli del pvalue
    abs_t_statistics = [abs(x) for x in welch_dict['all_t_values']]
    sorted_t_statistics = sorted(abs_t_statistics)
    x = np.linspace(0, 30000, 28547)
    plt.plot(x, sorted_t_statistics)
    plt.show()

    welch_t_bonferroni_genes_path = os.path.join(results_dir, "welch_t_bonferroni_genes.txt")
    fp = open(welch_t_bonferroni_genes_path, "w")
    for gene, t_value in zip(welch_dict['genes_b'], welch_dict['t_values_b']):
        fp.write("%s %f\n" % (gene, t_value))
    fp.close()

    # selected gene with bigger statistics value (more differentially expressed)
    # -> it is the same of p_value selected?
    # are the performances better?
    # perchè abbiamo scelto 15 come valore di threshold per il t_value?
    zipped = zip(df.columns, welch_dict['all_t_values'])  # creo lista di tuple (gene_name, t_value)
    selected_zipped = list(filter(lambda t: abs(t[1]) >= params['t_stat_threshold'],
                                  zipped))  # filtro la lista in base al valore di t_value in valore assoluto
    selected_genes, selected_t_statistics = map(list, zip(*selected_zipped))  # unzip and return two lists
    print(f'>> Number of most significant genes: {len(selected_genes)}')

    '''
    # Sort genes to respect p-value
    zipped = zip(welch_dict['genes_b'], welch_dict['p_values_b'])
    sorted_zipped = sorted(zipped, key=lambda t: t[1])
    sorted_p_values, sorted_genes = map(list, zip(*sorted_zipped))  # unzip

    # For the chosen method:
    df_reduced = df[sorted_genes]
    n_features = len(df_reduced.columns)  # update number of features
    df_reduced_0 = df_reduced.loc[df_reduced.index.str.endswith('_0')]
    df_reduced_1 = df_reduced.loc[df_reduced.index.str.endswith('_1')]
    '''


    # 2.a.2 Apply logarithmic transformation on gene expression data
    #       Description : x = Log(x+1), where x is the gene expression value
    print(f'\n[DGEA pre-processing] Logarithmic transformation on gene expression data:'
          f'\n>> Computing logarithmic transformation...')
    df = df.applymap(lambda x: math.log(x + 1, 10))


    # svm t rfe
    print("\n[DGEA svm-t-rfe]:")
    df_reduced = df[selected_genes]
    df_reduced_array = df_reduced.to_numpy()
    print(df_reduced_array)
    n_features = len(df_reduced.columns)  # update number of features
    genes_name = [gene for gene in df_reduced.columns]  # create list of gene names
    y = numpy.array([x[-1:] for x in df_reduced.index])  # create numpy array of class labels

    #print(y)

    ranked_genes = []
    ranked_t_statistics = []
    param_grid = dict(svm__C=[0.0001, 0.001, 0.01, 0.1, 1])

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    ranking_genes_file = str(params['alpha']) + "_" + str(params['t_stat_threshold']) + "_" + \
                         str(params['theta']) + "_" + str(params['cv_grid_search_rank']) + "_ranked_genes.txt"
    ranking_genes_path = os.path.join(results_dir, ranking_genes_file)

    c_values_file = str(params['alpha']) + "_" + str(params['t_stat_threshold']) + "_" + \
                    str(params['theta']) + "_" + str(params['cv_grid_search_rank']) + "_cvalues.txt"

    perform_grid_search = False
    c_values = []

    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    path_to_c_values = os.path.join(config_dir, c_values_file)
    if os.path.exists(path_to_c_values):
        fp = open(path_to_c_values, 'r')
        lines = fp.readlines()
        for i, c in enumerate(lines):
            if common.is_float(c):
                c_values.append(float(c))
            else:
                print(f'Line {i} is corrupt!')
                c_values.clear()
                break
        fp.close()

    if len(c_values) == 0:
        perform_grid_search = True

    # Rank
    i = 0
    pbar = tqdm(desc=">> Ranking genes...", total=df_reduced_array.shape[1], file=sys.stdout)
    while df_reduced_array.shape[1] > 0:
        # aggiunta ricerca del miglior C
        if perform_grid_search:
            pipe_grid = Pipeline([('svm', SVC(kernel='linear', class_weight='balanced'))])
            cv = StratifiedKFold(n_splits=params['cv_grid_search_rank'], shuffle=True, random_state=42)
            grid = GridSearchCV(estimator=pipe_grid, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=cv, refit=True)
            grid = grid.fit(df_reduced_array, y)
            C = grid.best_params_['svm__C']
            c_values.append(C)
            # best_model = grid.best_estimator_
            # weights = best_model.coef_[0]
        else:
            C = c_values[i]
        # dato il miglior C trovato con la grid search, faccio quello che facevamo prima
        pipe = Pipeline([('svm', SVC(kernel='linear', class_weight='balanced', C=C))])
        pipe = pipe.fit(df_reduced_array, y)
        model = pipe.named_steps['svm']
        #weights = [abs(x) for x in model.coef_[0]] # dal paper ho visto che considera sia weights che t_statistics in valore assoluto
        weights = model.coef_[0]
        ranking_scores = []
        weights_norm = norm(weights)
        selected_t_statistics_norm = norm(selected_t_statistics)
        for j in range(len(weights)):
            # divido per la norma del vettore -> fatto sia per weights che per selected_t_statistics. Corretto?!
            r = params['theta'] * (abs(weights[j] / weights_norm)) + (1 - params['theta']) * (
                    abs(selected_t_statistics[j] / selected_t_statistics_norm))
            ranking_scores.append(r)
        min_score = min(ranking_scores)  # sto prendendo il minimo

        indexes_to_remove = []
        for k in range(len(ranking_scores)):
            if ranking_scores[k] == min_score:  # rimuovo tutti i geni con score == min_score
                indexes_to_remove.append(k)

        for unwanted_index in sorted(indexes_to_remove, reverse=True):
            ranked_t_statistics.append(selected_t_statistics[unwanted_index])
            del selected_t_statistics[unwanted_index]
            df_reduced_array = np.delete(df_reduced_array, unwanted_index, 1)
            ranked_genes.append(genes_name[unwanted_index])
            del genes_name[unwanted_index]
            pbar.update(1)
        i += 1
    pbar.close()

    # salvo nel file i best C trovati
    if perform_grid_search:
        fp = open(path_to_c_values, "w")
        for c in c_values:
            fp.write("%f\n" % c)
        fp.close()

    ranked_genes.reverse()
    file_genes = open(ranking_genes_path, "w")
    for gene_name in ranked_genes:
        file_genes.write("%s\n" % gene_name)
    file_genes.close()


    # prendo gli ultimi top ranked valori
    top_ranked_genes = ranked_genes[:params['top_ranked']]  # ordino i geni dal primo top ranked all'ultimo top ranked. Corretto?
    top_selected_t = ranked_t_statistics[:-params['top_ranked'] + 1:-1]
    df_final = df_reduced[top_ranked_genes]
    accuracy = []
    # costruisco la param_grid per la ricerca dei migliori hyperparms
    if params['kernel'] == 'rbf':
        # if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma
        C_range = [0.01, 0.1, 1]
        gamma_range = [0.001, 0.01, 0.1, 'scale']
        param_grid = dict(svm__gamma=gamma_range, svm__C=C_range)
    else:
        C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = dict(svm__C=C_range)

    # calcolo l'accuratezza considerando prima solo il primo top ranked gene, poi solo i primi due, etc..
    # fino ad arrivare a considerare tutti i top ranked genes
    # ho tolto la parte in cui calcolo i ranking scores per capire quale gene eliminare.
    # Pensandoci, mi sembra che non serva rifare il calcolo per capire quale gene togliere
    for num_selected in tqdm(range(1, params['top_ranked']+1), desc=">> Computing accuracies on top ranked...", file=sys.stdout):
        # invece di fare un leave one out, faccio un k fold. Quindi teniamo fuori dal training più di un elemento.
        # il tutto diventa più veloce. Il leave one out si fa quando si hanno molti pochi dati.
        # Forse nel nostro caso non è necessario?! (Da pensare)
        selected_features = top_ranked_genes[:num_selected]
        df_selected_array = df_final[selected_features].to_numpy()
        # stratified enforce the class distribution in each split of the data to match the distribution in the complete training dataset
        cv_outer = StratifiedKFold(n_splits=params['cv_outer'], shuffle=True, random_state=42)
        outer_results = []
        for train_ix, test_ix in cv_outer.split(df_selected_array, y):
            X_train, X_test = df_selected_array[train_ix, :], df_selected_array[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            # configure the cross-validation procedure
            # aggiunta ricerca del miglior C: questo rallenta il tutto ma mi sembra ancora accettabile
            pipe_grid = Pipeline([('svm', SVC(kernel=params['kernel'], class_weight='balanced'))])
            cv_inner = StratifiedKFold(n_splits=params['cv_grid_search_acc'], shuffle=True, random_state=42)
            grid = GridSearchCV(estimator=pipe_grid, param_grid=param_grid, scoring='accuracy', cv=cv_inner, n_jobs=-1, refit=True)
            grid = grid.fit(X_train, y_train)
            # dato il miglior C trovato con la grid search, faccio quello che facevamo prima
            if params['kernel'] == 'rbf':
                
                #print("C")
                #print(grid.best_params_['C'])
                #print("gamma")
                #print(grid.best_params_['gamma'])
                
                pipe = Pipeline([
                                  ('svc', SVC(kernel=params['kernel'], class_weight='balanced', C=grid.best_params_['svm__C'], gamma=grid.best_params_['svm__gamma']))])
            else:
                pipe = Pipeline([
                                 ('svc', SVC(kernel=params['kernel'], class_weight='balanced', C=grid.best_params_['svm__C']))])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, pred)
            #print(acc)
            outer_results.append(acc)

        global_acc = mean(outer_results)  # considero l'accuratezza globale come la media delle accuratezze ottenute durante il k-fold
        accuracy.append(global_acc)
        #print("global acc")

    x = np.arange(1, len(accuracy)+1)

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_ylim(bottom=0.992)
    ax.set_xlim(0, len(x))
    plt.plot(x, accuracy)
    plt.xlabel("Number of top ranked genes")
    plt.ylabel("Prediction accuracy")
    plt.margins(x=0)
    plt.show()
    plt.close()

    # Typically, 50–100 genes have been selected in previous studies

