import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.genes.features_selection_methods import common


def genes_extraction_welch_t_pca(df, params):
    # Remove genes with median = 0
    print("[DGEA pre-processing] Removing genes with median = 0:")
    df, removed_genes = common.remove_genes_with_median_0(df)
    n_features = len(df.columns)  # update number of features

    print(f'>> Removed genes: {removed_genes}'
          f'\n>> Number of genes removed: {len(removed_genes)}'
          f'\n>> Number of genes remained: {n_features}')

    # 2.a.2 Apply logarithmic transformation on gene expression data
    #       Description : x = Log(x+1), where x is the gene expression value
    print(f'\n[DGEA pre-processing] Logarithmic transformation on gene expression data:'
          f'\n>> Computing logarithmic transformation...')
    df_log_transformed = df.applymap(lambda x: math.log(x + 1, 10))

    print(f'>> First 10 values before transformation:\n{df.head(10)}')
    print(f'>> First 10 values after transformation:\n{df_log_transformed.head(10)}')
    
    # Separate patients data frame in normal data frame and tumor data frame
    df_0_log_transformed = df_log_transformed.loc[df_log_transformed.index.str.endswith('_0')]
    df_1_log_transformed = df_log_transformed.loc[df_log_transformed.index.str.endswith('_1')]

    '''
    # 2.a.3 Check for normality of our data
    #       For each gene, I check the normality distribution
    #       of the 2 groups of data (healthy data and diseased data)
    alpha = 0.05
    # Anderson test
    print("\n[DGEA pre-processing] Anderson test:")
    a_normal_genes = anderson_normality_test(df_log_transformed, df_1_log_transformed)

    # Shapiro test
    print("\n[DGEA pre-processing] Shapiro test:")
    s_normal_genes = shapiro_normality_test(df_0_log_transformed, df_1_log_transformed, alpha)

    # Normal test
    print("\n[DGEA pre-processing] Normal test:")
    n_normal_genes = normal_test(df_0_log_transformed, df_1_log_transformed, alpha)

    # Results
    print(f'\n>> Number of genes with both healthy and diseased patient groups normally distributed:'
          f'\n>> Result for Anderson test: {len(a_normal_genes)} (over {n_features})'
          f'\n>> Result for Shapiro test: {len(s_normal_genes)} (over {n_features})'
          f'\n>> Result for Normal test: {len(n_normal_genes)} (over {n_features})')
    '''

    # 2.b Statistical test
    '''
    # 2.b.1 Mann–Whitney U test
    alpha = 0.05
    print("\n[DGEA statistical test] Mann–Whitney U test statistics:")
    m_reduced_genes = mann_whitney_u_test(df_0, df_1, alpha)

    print(f'>> Reduced genes: {m_reduced_genes}'
          f'\n>> Number of selected genes (features): {len(m_reduced_genes)}')
    '''

    # 2.b.2 Welch t test
    print("\n[DGEA statistical test] Welch t-test statistics:")
    w_reduced_genes, _, _, _, _, _, _ = common.welch_t_test(df_0_log_transformed, df_1_log_transformed, params['alpha'])

    df_reduced = df[w_reduced_genes]
    n_features = len(df_reduced.columns)  # update number of features

    df_reduced_0 = df_reduced.loc[df_reduced.index.str.endswith('_0')]
    df_reduced_1 = df_reduced.loc[df_reduced.index.str.endswith('_1')]

    print(f'>> Reduced genes: {w_reduced_genes}'
          f'\n>> Number of selected genes (features): {n_features}'
          f'\n>> Shape of reduced normal dataset: {df_reduced_0.shape}'
          f'\n>> Shape of reduced tumor dataset: {df_reduced_1.shape}')

    # 2.d Per-class feature histograms, for the genes selected by Welch t test
    print("\n[DGEA visualization step] Per-class feature histograms, for first 30 genes selected by Welch t test")
    common.plot_histograms(df_reduced_0, df_reduced_1, colors=['orange', 'blue'])

    # 2.c Per-class feature histograms, for the genes discarded by Welch t test
    print("\n[DGEA visualization step] Per-class feature histograms, for first 30 genes discarded by Welch t test")
    w_not_selected_genes = df.columns.difference(w_reduced_genes)
    df_no_expr = df[w_not_selected_genes]
    common.plot_histograms(df_no_expr.loc[df_no_expr.index.str.endswith('_0')],
                              df_no_expr.loc[df_no_expr.index.str.endswith('_1')], colors=['orange', 'green'])


    # 2.d hierarchical clustering
    # TODO
    common.plot_hierarchical_clustering(df_reduced_0)
    common.plot_hierarchical_clustering(df_reduced_1)

    # 2.e PCA
    print("\n[DGEA dimensionality reduction] Principal component analysis (PCA)")
    # 2.e.1 Undersample majority class and divide data in training and test data

    sample_size = 200
    indices_undersampling, X_train_1 = common.random_undersample(df_reduced_1, sample_size)
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_train_1, X_train_1.index,
                                                                                      train_size=0.50, random_state=0)
    X_valid_1, X_test_1, y_valid_1, y_test_1 = train_test_split(X_valid_1, y_valid_1,
                                                                                      train_size=0.20, random_state=0)
    X_train_0, X_valid_0, y_train_0, y_valid_0 = train_test_split(df_reduced_0, df_reduced_0.index,
                                                                                      train_size=0.50, random_state=0)
    X_valid_0, X_test_0, y_valid_0, y_test_0 = train_test_split(X_valid_0, y_valid_0,
                                                                                      train_size=0.20, random_state=0)
    X_train = X_train_1.append(X_train_0)
    X_valid = X_valid_1.append(X_valid_0)
    X_test = X_test_1.append(X_test_0)
    y_train = y_train_1.append(y_train_0)
    y_valid = y_valid_1.append(y_valid_0)
    y_test = y_test_1.append(y_test_0)

    print(X_train.shape)
    print(type(X_train))
    # 2.e.2 Standardize features by removing the mean and scaling to unit variance
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.transform(X_valid)
    X_test = sc.transform(X_test)

    # 2.e.3 Perform the dimensionality reduction onto dataset with PCA (principal component Analysis)
    #       Description : n_components = 0.95 indicates that the amount of variance that needs
    #                     to be explained is greater than the percentage specified by n_components
    pca = PCA(n_components=params['percentage_variance'])
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)
    print(f'>> Shape of training set after PCA: {X_train_pca.shape}')
    print(f'>> Shape of validation set after PCA: {X_valid_pca.shape}')
    print(f'>> Shape of test set after PCA: {X_test_pca.shape}')

    # 2.e.4 Plot the explained variance as a function of the number of dimensions
    print(f'>> Plotting the explained variance as a function of the number of dimensions')
    common.plot_variance_vs_num_components(X_train_pca)

    # Plot first vs second principal component
    common.plot_pca_2D(X_train_pca, y_train)

    # Scree plot
    common.scree_plot(pca)

    # Biplot
    #biplot(X_train_pca[:, 0:2], np.transpose(pca.components_[0:2, :]))


