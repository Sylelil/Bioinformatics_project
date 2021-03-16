import argparse
import os
import sys
from os import path
from pathlib import Path

from sklearn.model_selection import train_test_split
import numpy as np
from src.genes import methods
from src.genes.features_selection_methods.pca import genes_extraction_pca
from src.genes.features_selection_methods.svm_t_rfe import genes_selection_svm_t_rfe
from src.genes.features_selection_methods.welch_t import genes_selection_welch_t
from src.genes.features_selection_methods.welch_t_pca import genes_extraction_welch_t_pca


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        help='Configuration file name',
                        required=True,
                        type=str)

    parser.add_argument('--method',
                        help='Feature extraction method',
                        choices=['pca', 'svm_t_rfe', 'welch_t', 'welch_t_pca'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    if not path.exists(args.cfg) or (not path.isfile(args.cfg)):
        sys.stderr.write("Invalid path for config file")
        exit(2)

    # Read configuration file
    params = methods.read_config_file(args.cfg, args.method)

    path_genes = Path('datasets') / 'genes'
    if not os.path.exists(path_genes):
        sys.stderr.write(f'{path_genes} does not exists')
        exit(2)

    print("Reading gene expression data:")
    df_normal, df_tumor = methods.read_gene_expression_data(path_genes)  # normal = 0, tumor = 1

    print("\nExploratory analysis:")
    # Compute number of samples
    n_samples_normal = len(df_normal)
    n_samples_tumor = len(df_tumor)
    print(f'>> Tumor samples: {n_samples_tumor}\n>> Normal samples: {n_samples_normal}')

    # Compute number of features
    df_patients = df_normal.append(df_tumor, sort=False)  # Merge normal data frame with tumor data frame
    n_features = len(df_patients.columns)
    print(f">> Number of features (genes): {n_features}")
    print(df_patients)

    # Evaluate normality by skewness and kourt
    n_skew_pos, n_skew_neg, n_kurt_1, n_kurt_2 = methods.eval_asymmetry_and_kurt(df_patients)

    print("Percentage of genes with asymmetric distribution (verso sx): %.3f" % (100 * (n_skew_pos / n_features)))
    print("Percentage of genes with asymmetric distribution (verso dx): %.3f" % (100 * (n_skew_neg / n_features)))
    print("Percentage of genes with platykurtic distribution: %.3f" % (100 * (n_kurt_2 / n_features)))
    print("Percentage of genes with leptokurtic distribution: %.3f" % (100 * (n_kurt_1 / n_features)))

    # Grafico a torta di skew and kurtosys
    # TODO

    # divide dataset in training and test
    '''
    y = np.array([x[-1:] for x in df_patients])
    X_train, X_test, y_train, y_test = train_test_split(df_patients, y, train_size=0.70, random_state=0)
    print(y_train)
    '''

    print("\nDifferentially gene expression analysis [DGEA]")
    if args.method == 'pca':
        genes_extraction_pca(df_patients, params)
    elif args.method == 'welch_t_pca':
        genes_extraction_welch_t_pca(df_patients, params)
    elif args.method == 'welch_t':
        genes_selection_welch_t(df_patients, params)
    elif args.method == 'svm_t_rfe':
        results_dir = Path('results') / 'genes'
        config_dir = Path('config') / 'genes'
        if not os.path.exists(config_dir):
            os.mkdir(config_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        genes_selection_svm_t_rfe(df_patients, params, results_dir, config_dir)
    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == "__main__":
    main()
