import os
import argparse
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

def compute_scaling_pca(params, train_filepath, val_filepath, test_filepath):
    x_train_pca_path = Path('assets') / 'concatenated_pca' / 'x_train.npy'
    y_train_pca_path = Path('assets') / 'concatenated_pca' / 'y_train.npy'
    x_val_pca_path = Path('assets') / 'concatenated_pca' / 'x_val.npy'
    y_val_pca_path = Path('assets') / 'concatenated_pca' / 'y_val.npy'
    x_test_pca_path = Path('assets') / 'concatenated_pca' / 'x_test.npy'
    y_test_pca_path = Path('assets') / 'concatenated_pca' / 'y_test.npy'

    if os.path.exists(Path('assets') / 'concatenated_pca'):
        print('>> Reading files with scaled and pca data previously computed...')
        X_train = np.load(x_train_pca_path)
        y_train = np.load(y_train_pca_path)
        X_val = np.load(x_val_pca_path)
        y_val = np.load(y_val_pca_path)
        X_test = np.load(x_test_pca_path)
        y_test = np.load(y_test_pca_path)

    else:
        os.mkdir(Path('assets') / 'concatenated_pca')
        batchsize = params['preprocessing']['batchsize']

        print(">> Fitting scaler...")
        scaler = StandardScaler()
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            scaler.partial_fit(X_train_chunk)

        ipca = IncrementalPCA(n_components=params['pca']['n_components'])
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []
        print(">> Transforming train data with scaler and fitting incremental pca...")
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            X_train_chunk_scaled = scaler.transform(X_train_chunk)
            ipca.partial_fit(X_train_chunk_scaled)
        print(">> Transforming train data with incremental pca...")
        for chunk in tqdm(pd.read_csv(train_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_train_chunk = chunk.iloc[:, :-1]
            y_train_chunk = chunk['label']
            X_train_chunk_scaled = scaler.transform(X_train_chunk)
            X_train_chunk_ipca = ipca.transform(X_train_chunk_scaled)
            X_train.extend(X_train_chunk_ipca)
            y_train.extend(y_train_chunk)
        print(">> Transforming validation data with incremental pca...")
        for chunk in tqdm(pd.read_csv(val_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_val_chunk = chunk.iloc[:, :-1]
            y_val_chunk = chunk['label']
            X_val_chunk_scaled = scaler.transform(X_val_chunk)
            X_val_chunk_ipca = ipca.transform(X_val_chunk_scaled)
            X_val.extend(X_val_chunk_ipca)
            y_val.extend(y_val_chunk)
        print(">> Transforming test data with incremental pca...")
        for chunk in tqdm(pd.read_csv(test_filepath, chunksize=batchsize, iterator=True, dtype='float64')):
            X_test_chunk = chunk.iloc[:, :-1]
            y_test_chunk = chunk['label']
            X_test_chunk_scaled = scaler.transform(X_test_chunk)
            X_test_chunk_ipca = ipca.transform(X_test_chunk_scaled)
            X_test.extend(X_test_chunk_ipca)
            y_test.extend(y_test_chunk)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        print(">> Saving computed features on files in assets/concatenated_pca/ folder...")
        np.save(x_train_pca_path, X_train)
        np.save(y_train_pca_path, y_train)
        np.save(x_val_pca_path, X_val)
        np.save(y_val_pca_path, y_val)
        np.save(x_test_pca_path, X_test)
        np.save(y_test_pca_path, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test