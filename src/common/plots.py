from pathlib import Path
import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def __plot_loss(history, label, n):
    """
       Description: Private function. Plot train and validation loss.
       :param history: model history.
       :param label: label string.
       :param n: color number.
    """
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend(loc='upper right')
    plt.grid(True)


def __plot_metrics(history):
    """
       Description: Private function. Plot train and validation metrics scores for NN.
       :param history: model history.
    """
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1.1])

        plt.title(f'Model {name}')
        plt.legend()


def plot_train_val_results(history, save_path):
    """
       Description: Plot train and validation loss and metrics scores.
       :param history: model history.
       :param save_path: path to save figures.
    """
    __plot_loss(history, 'loss', 0)
    plt.savefig(Path(save_path) / 'train_val_loss')
    plt.figure()
    __plot_metrics(history)
    plt.savefig(Path(save_path) / 'train_val_metrics')
    plt.figure()


def plot_cm(labels, predictions):
    """
       Description: Plot Confusion Matrix.
       :param labels: ground truth labels.
       :param predictions: predicted labels.
    """
    cm = metrics.confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc(name, labels, predictions, **kwargs):
    """
       Description: Plot ROC curve.
       :param name: title.
       :param labels: ground truth labels.
       :param predictions: predicted labels.
       :param **kwargs: plot arguments.
    """
    fp, tp, _ = metrics.roc_curve(labels, predictions)
    auc = metrics.roc_auc_score(labels, predictions)

    plt.plot(fp, tp, label=f'{name}, AUC={auc}', linewidth=2, **kwargs)
    plt.xlabel('False positive rate (Positive label: 1)')
    plt.ylabel('True positive rate (Positive label: 1)')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title('ROC curve')
    plt.legend(loc='lower right')


def plot_prc(name, labels, predictions, **kwargs):
    """
       Description: Plot Precision-Recall curve.
       :param name: title.
       :param labels: ground truth labels.
       :param predictions: predicted scores.
       :param **kwargs: plot arguments.
    """
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
    ap = metrics.average_precision_score(labels, predictions)

    plt.plot(recall, precision, label=f'{name}, AP={ap}', linewidth=2, **kwargs)
    plt.xlabel('Recall (Positive label: 1)')
    plt.ylabel('Precision (Positive label: 1)')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower right')


def plot_explained_variance(explained_variance_ratio, path_to_save, n_components):
    print(">> Plotting individual and cumulative explained variance...")
    plt.figure()
    plt.plot(np.cumsum(explained_variance_ratio), label='Cumulative explained variance', linewidth=2, marker='.')
    plt.plot(explained_variance_ratio, label='Individual explained variance', linewidth=2, marker='.')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Number of components')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(Path(path_to_save) / f'explained_variance_{n_components}')
    plt.show()
    print('>> Done.')


def plot_pca(path_to_save, X, y):
    """
        Description: Plot data in 2D with PCA, by considering the first 2 principal components

        :param path_to_save: path to save figure

        :param X: 2D-numpy array, shape = [n_samples, n_features],
            where n_samples is the number of samples and n_features is the number of SCALED features.

        :param y: array-like, shape = [n_samples]
            labels (0/1)
    """
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X)
    print('>> Cumulative explained variation for 2 principal components: {}'.format(
        np.sum(pca.explained_variance_ratio_)))

    df_pca = pd.DataFrame(
        dict(pca_one=pca_results[:, 0], pca_two=pca_results[:, 1], y=y))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="pca_one", y="pca_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_pca,
        legend="full",
        alpha=0.9,
    )
    plt.xlabel('pca_one')
    plt.ylabel('pca_two')
    plt.savefig(path_to_save)
    plt.show()


def plot_pca_3D(path_to_save, X, y):
    """
        Description: Plot data in 3D with PCA, by considering the first 3 principal components

        :param path_to_save: path to save figure

        :param X: 2D-numpy array, shape = [n_samples, n_features],
            where n_samples is the number of samples and n_features is the number of SCALED features.

        :param y: array-like, shape = [n_samples]
            labels (0/1)

    """
    pca = PCA(n_components=3, random_state=42)
    pca_results = pca.fit_transform(X)
    print('>> Cumulative explained variation for 3 principal components: {}'.format(
        np.sum(pca.explained_variance_ratio_)))

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    scatter = ax.scatter(
        xs=pca_results[:, 0],
        ys=pca_results[:, 1],
        zs=pca_results[:, 2],
        c=y,
        cmap='tab10'
    )
    legend = ax.legend(*scatter.legend_elements())
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    ax.add_artist(legend)
    plt.savefig(path_to_save)
    plt.show()


def plot_tsne_pca(path_to_save, X, y):
    """
        Description: Plot data in 2D with t-SNE, by first applying PCA on the 50 principal components

        :param path_to_save: path to save figure

        :param X: 2D-numpy array, shape = [n_samples, n_features],
            where n_samples is the number of samples and n_features is the number of SCALED features.

        :param y: array-like, shape = [n_samples]
            labels (0/1)
    """
    pca_50 = PCA(n_components=50, random_state=42)
    pca_result_50 = pca_50.fit_transform(X)
    print('>> Cumulative explained variation for 50 principal components: {}'.format(
        np.sum(pca_50.explained_variance_ratio_)))

    tsne = TSNE(n_components=2, random_state=42)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    df_tsne_pca = pd.DataFrame(
        dict(tsne_pca50_one=tsne_pca_results[:, 0], tsne_pca50_two=tsne_pca_results[:, 1], y=y))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="tsne_pca50_one", y="tsne_pca50_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_tsne_pca,
        legend="full",
        alpha=0.9,
    )
    plt.xlabel('tsne_pca50_one')
    plt.ylabel('tsne_pca50_two')
    plt.savefig(path_to_save)
    plt.show()


def plot_tsne(path_to_save, X, y):
    """
        Description: Plot data in 2D with t-SNE

        :param path_to_save: path to save figure

        :param X: 2D-numpy array, shape = [n_samples, n_features],
            where n_samples is the number of samples and n_features is the number of SCALED features.

        :param y: array-like, shape = [n_samples]
            labels (0/1)
    """

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(
        dict(tsne_one=tsne_results[:, 0], tsne_two=tsne_results[:, 1], y=y))
    plt.figure(figsize=(16, 7))
    sns.scatterplot(
        x="tsne_one", y="tsne_two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_tsne,
        legend="full",
        alpha=0.9,
    )
    plt.xlabel('tsne_one')
    plt.ylabel('tsne_two')
    plt.savefig(path_to_save)
    plt.show()


def plot_tsne_3D(path_to_save, X, y):
    """
        Description: Plot data in 3D with t-SNE

        :param path_to_save: path to save figure

        :param X: 2D-numpy array, shape = [n_samples, n_features],
            where n_samples is the number of samples and n_features is the number of SCALED features.

        :param y: array-like, shape = [n_samples]
             labels (0/1)
     """

    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(X)

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    scatter = ax.scatter(
        xs=tsne_results[:, 0],
        ys=tsne_results[:, 1],
        zs=tsne_results[:, 2],
        c=y,
        cmap='tab10'
    )
    legend = ax.legend(*scatter.legend_elements())
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_zlabel('tsne-three')
    ax.add_artist(legend)
    plt.savefig(path_to_save)
    plt.show()


def plot_2D_svm_decision_boundary(path_to_save, clf, X_train, y_train, X_test, y_test):
    """
        Description: show 2D decision boundary for SVM fit on first 2 features

        :param path_to_save: path to save figure

        :param clf: SVM classifier

        :param X_train: 2D-numpy array, shape = [n_samples, 2],
               where n_samples is the number of training samples and 2 is the number of SCALED features.

        :param y_train: array-like, shape = [n_samples]
                training labels (0/1)

        :param X_test: 2D-numpy array, shape = [n_samples, 2],
               where n_samples is the number of test samples and 2 is the number of SCALED features.

        :param y_test: array-like, shape = [n_samples]
                test labels (0/1)
    """

    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig(path_to_save)
    plt.show()


def plot_2D_svm_decision_boundary_integration(path_to_save, clf, X_train, y_train, X_test, y_test):
    """
        Description: show 2D decision boundary for SVM fit on first 2 features

        :param path_to_save: path to save figure

        :param clf: SVM classifier

        :param X_train: 2D-numpy array, shape = [n_samples, 2],
               where n_samples is the number of training samples and 2 is the number of SCALED features.

        :param y_train: array-like, shape = [n_samples]
                training labels (0/1)

        :param X_test: 2D-numpy array, shape = [n_samples, 2],
               where n_samples is the number of test samples and 2 is the number of SCALED features.

        :param y_test: array-like, shape = [n_samples]
                test labels (0/1)
    """

    h = .02  # step size in the mesh
    x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5, max(X_train[:, 0].max(), X_test[:, 0].max()) + .5
    y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5, max(X_train[:, 1].max(), X_test[:, 1].max()) + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#FFFF00'])
    cm_bright2 = ListedColormap(['#00A000', '#0000FF'])

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright2, edgecolors='k')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig(path_to_save)
    plt.show()