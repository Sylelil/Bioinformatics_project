from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn import metrics
import numpy as np

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

'''
def plot_loss2(model_history):
    """
       Description: Plot loss.
       :param model_history: model history.
    """
    train_loss=[value for key, value in model_history.items() if 'loss' in key.lower()][0]
    valid_loss=[value for key, value in model_history.items() if 'loss' in key.lower()][1]
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_loss, '--', color=color, label='Train Loss')
    ax1.plot(valid_loss, color=color, label='Valid Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper left')
    plt.title('Model Loss')
    plt.show()
'''

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
       Description: Private function. Plot train and validation metrics scores.
       :param history: model history.
    """
    metrics = ['loss', 'prc', 'precision', 'recall']
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
    sns.heatmap(cm, annot=True, fmt="d")
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

    plt.plot(fp, tp, label=f'{name}, auc={auc}', linewidth=2, **kwargs)
    plt.xlabel('False positive rate (Positive label: 1)')
    plt.ylabel('True positive rate (Positive label: 1)')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc=0)


def plot_prc(name, labels, predictions, **kwargs):
    """
       Description: Plot Precision-Recall curve.
       :param name: title.
       :param labels: ground truth labels.
       :param predictions: predicted labels.
       :param **kwargs: plot arguments.
    """
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall (Positive label: 1)')
    plt.ylabel('Precision (Positive label: 1)')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


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



