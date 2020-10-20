import pandas as pd
#import tensorflow as tf
import os
import sys
from scipy import stats
import numpy as np


def main():

    path_ge = "C:\\Users\\rosee\\workspace_Polito\\git\\Bioinformatics_project\\assets\\final_genes\\"
    path_example_file = "C:\\Users\\rosee\\workspace_Polito\\git\\Bioinformatics_project\\assets\\final_genes\\0a94eecf-4db2-4846-8383-c83ff02e4a9f_1.txt"

    # Number of samples:
    n_samples_0 = sum(1 for x in os.listdir(path_ge) if x.endswith("_0.txt"))
    n_samples_1 = sum(1 for x in os.listdir(path_ge) if x.endswith("_1.txt"))
    print(f'>> Tumor samples: {n_samples_1}\n>> Normal samples: {n_samples_0}')

    # Number of features:
    with open(path_example_file) as f:
        n_features = sum(1 for _ in f)
    print(f">> Number of features (genes): {n_features}")

    # Read samples:
    df_0 = pd.DataFrame()
    df_1 = pd.DataFrame()
    i=0
    for file_name in os.listdir(path_ge):
        file_path = os.path.join(path_ge,file_name)
        with open(file_path) as f:
            print(f"Reading file {i}")
            patient_df = pd.read_csv(f, sep="\t", header=None, index_col=0, names=[file_name.replace(".txt","")])
            patient_df = pd.DataFrame.transpose(patient_df)
            if file_name.endswith("_0.txt"):
                df_0 = df_0.append(patient_df)
            else:
                df_1 = df_1.append(patient_df)
        i=i+1

    # T-test:
    alpha = 0.05
    reduced_genes = []
    print("Computing t-test statistics")
    for gene in df_0.columns:
        tvalue, pvalue = stats.ttest_ind(np.array(df_0[gene].tolist()), np.array(df_1[gene].tolist()), equal_var=False, nan_policy='omit')
        if pvalue <= alpha/n_features and not np.isnan(pvalue):
            reduced_genes.append(gene)
    print(f'>> Number of selected genes: {len(reduced_genes)}')

    df_reduced_0 = df_0[reduced_genes]
    df_reduced_1 = df_1[reduced_genes]

    print(f'>> Shape of reduced normal dataset: {df_reduced_0.shape}')
    print(f'>> Shape of reduced tumor dataset: {df_reduced_1.shape}')


if __name__ == "__main__":
    main()

