from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
import numpy as np


def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=-1)
    #labels = np.argmax(pred.label_ids, axis=-1)
    labels = pred.label_ids
    print(pred.label_ids.shape, pred.predictions.shape)
    print(pred.predictions)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision-macro": precision_score(labels, predictions, average="macro"),
        "recall-macro": recall_score(labels, predictions, average="macro"),
        "f1-macro": f1_score(labels, predictions, average="macro")
    }
