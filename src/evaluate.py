"""
Evaluation utilities for flower image classification models.

Provides:
- plot_learning_curves : accuracy and loss curves from a Keras History object
- evaluate_model       : confusion matrix and classification report on a dataset
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def plot_learning_curves(history, title_prefix: str = "Model") -> None:
    """
    Plot training and validation accuracy and loss curves.

    Args:
        history    : Keras History object returned by model.fit().
        title_prefix: Label used in the plot titles (e.g. "CNN v1", "DenseNet121").
    """
    # --- Accuracy ---
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title(f"{title_prefix} — Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Loss ---
    plt.figure()
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title(f"{title_prefix} — Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(model, val_ds, class_names: list[str]) -> None:
    """
    Print the confusion matrix and classification report for a trained model.

    Args:
        model      : Trained Keras model.
        val_ds     : Validation tf.data.Dataset (batched, prefetched).
        class_names: Ordered list of class label strings.
    """
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
