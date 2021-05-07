from sklearn import metrics
from tensorflow import keras

''' Metrics for shallow classification: '''
METRICS_skl = [
    metrics.accuracy_score,
    metrics.matthews_corrcoef,
    metrics.average_precision_score,
    metrics.f1_score,
    metrics.precision_score,
    metrics.recall_score,
    metrics.roc_auc_score,
]


''' Metrics for nn classification: '''
METRICS_keras = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

