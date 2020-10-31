import pandas as pd
import os
import sys
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hier
import pylab
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def plot_hierarchical_clustering(df):
    D = pdist(df) # distance matrix

    # Compute and plot dendrogram.
    fig = pylab.figure()
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    Y = hier.linkage(D, method='centroid')
    Z = hier.dendrogram(Y, orientation='right')
    axdendro.set_xticks([])
    axdendro.set_yticks([])

    print('-----------D-----------')
    print(D.shape)
    print(D.ind)
    print('-----------Y-----------')
    print(Y.shape)
    print('-----------Z-----------')
    print(Z.shape)

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.8])
    index = Z['leaves']
    D = D[index, :]
    D = D[:, index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    pylab.colorbar(im, cax=axcolor)

    # Display and save figure.
    fig.show()
    #fig.savefig('dendrogram.png')

def random_undersample(df, size):
    random_indices = np.random.choice(df.index, size, replace=False)
    df_undersampled = df.loc([[random_indices]])
    return random_indices, df_undersampled

def main():

    #path_ge = "C:\\Users\\rosee\\workspace_Polito\\git\\Bioinformatics_project\\assets\\final_genes\\"
    #path_example_file = "C:\\Users\\rosee\\workspace_Polito\\git\\Bioinformatics_project\\assets\\final_genes\\0a94eecf-4db2-4846-8383-c83ff02e4a9f_1.txt"
    path_ge = sys.argv[1]

    # Number of samples:
    n_samples_0 = sum(1 for x in os.listdir(path_ge) if x.endswith("_0.txt"))
    n_samples_1 = sum(1 for x in os.listdir(path_ge) if x.endswith("_1.txt"))
    print(f'>> Tumor samples: {n_samples_1}\n>> Normal samples: {n_samples_0}')

    # Number of features:
    # with open(path_example_file) as f:
    #    n_features = sum(1 for _ in f)
    # print(f">> Number of features (genes): {n_features}")

    # Read samples:
    df_0 = pd.DataFrame()
    df_1 = pd.DataFrame()

    print(f'Reading gene expression data...')
    itr=0
    for file_name in tqdm(os.listdir(path_ge)):
        file_path = os.path.join(path_ge,file_name)
        with open(file_path) as f:
            patient_df = pd.read_csv(f, sep="\t", header=None, index_col=0, names=[file_name.replace(".txt","")])
            # Calculate number of features:
            if itr == 0:
                n_features = len(patient_df)
            itr=1
            patient_df = pd.DataFrame.transpose(patient_df)
            if file_name.endswith("_0.txt"):
                df_0 = df_0.append(patient_df)
            else:
                df_1 = df_1.append(patient_df)

            #print(f"Reading file {itr}")

    print(f">> Number of features (genes): {n_features}")

    '''
    # test normality
    normal_like_genes = []
    print("Checking for normality...")
    for gene in df_0.columns:
        #print(gene)
        #statistic_0, pvalue_0 = stats.shapiro(df_0[gene].tolist())
        statistic_0, pvalue_0 = stats.normaltest(df_0[gene].tolist())
        #print(f'statistic_0: {statistic_0}\npvalue_0: {pvalue_0}')

        #statistic_1, pvalue_1 = stats.shapiro(df_1[gene].tolist())
        statistic_1, pvalue_1 = stats.normaltest(df_1[gene].tolist())
        #print(f'statistic_1: {statistic_1}\npvalue_1: {pvalue_1}')
        #print('\n')

        #if pvalue_0 > 0.05 and pvalue_1 > 0.05: # shapiro test
        if pvalue_0 < 1e-3 and pvalue_1 < 1e-3:
            normal_like_genes.append(gene)

    print(f'normal like genes:\n{normal_like_genes}')
    print(len(normal_like_genes))
    '''

    # T-test:
    alpha = 0.05
    reduced_genes = []

    print("Computing t-test statistics")
    for gene in tqdm(df_0.columns):
        tvalue, pvalue = stats.ttest_ind(np.array(df_0[gene].tolist()), np.array(df_1[gene].tolist()), equal_var=False, nan_policy='omit')
        if not np.isnan(pvalue) and pvalue <= alpha/n_features:
            reduced_genes.append(gene)

    print(f'>> Number of selected genes: {len(reduced_genes)}')

    df_reduced_0 = df_0[reduced_genes]
    df_reduced_1 = df_1[reduced_genes]

    print(f'>> Shape of reduced normal dataset: {df_reduced_0.shape}')
    print(f'>> Shape of reduced tumor dataset: {df_reduced_1.shape}')

    # Hierarchical clustering for visualization:
    #plot_hierarchical_clustering(df_reduced_0)
    #plot_hierarchical_clustering(df_reduced_1)


    # Divide data in training and test data
    X_train_0, X_test_0 = train_test_split(df_reduced_0,  test_size=0.2, random_state=0)
    X_train_1, X_test_1 = train_test_split(df_reduced_1, test_size=0.2, random_state=0)

    X_train = X_train_0.append(X_train_1, sort=False)
    X_test = X_test_0.append(X_test_1, sort=False)

    # Standardize features by removing the mean and scaling to unit variance
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Perform the dimensionality reduction onto dataset with PCA (principal component Analysis) using N features

    # Plot PC

    # Random undersampling of majority class (tumor)
    sample_size = 100
    indices_undersampling, X_train_1_undersampled = random_undersample(X_train_1, sample_size)
    # usare i restanti sample come X_test_1???????


if __name__ == "__main__":
    main()

