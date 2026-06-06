"""
Evaluation utilities: metrics, confusion matrix, Grad-CAM visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import seaborn as sns


# ── TODO 12 ──────────────────────────────────────────────────────────────────
def evaluate_model(model: keras.Model, test_data) -> dict:
    """
    Evaluate model on test_data and return a metrics dictionary.

    Tasks:
        1. Generate predictions with model.predict(test_data).
        2. Convert sigmoid probabilities to binary labels (threshold 0.5).
        3. Collect true labels from test_data.
        4. Compute and return a dict with keys:
               'accuracy', 'precision', 'recall', 'f1', 'auc',
               'confusion_matrix', 'classification_report'
           Use sklearn.metrics for precision, recall, f1 (average='binary').

    Args:
        model:     Trained keras.Model.
        test_data: Generator or tf.data.Dataset.

    Returns:
        dict of evaluation metrics.
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 12: implement evaluate_model()")
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    # Generate prediction probabilities
    # y_probs = model.predict(test_data).squeeze()
    y_probs = model.predict(test_data, verbose=1).squeeze()

    # Convert probabilities to binary predictions
    y_pred = (y_probs >= 0.5).astype(int)
    
    # Reset the generator to start from the beginning
    # test_data.reset() 
    
    # If test_data is a ImageDataGenerator, it has a 'classes' attribute with true labels
    # y_true = test_data.classes 

    # Collect true labels
    y_true = np.concatenate(
        [y for _, y in test_data]
    )

    # Return all required metrics
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_probs),
        # "roc_curve": roc_curve(y_true, y_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }
    
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 13 ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list = None) -> None:
    """
    Plot a labelled confusion matrix heatmap with seaborn.

    Tasks:
        1. Use seaborn.heatmap with annot=True, fmt='d', cmap='Blues'.
        2. Label axes with class_names (default ["NORMAL", "PNEUMONIA"]).
        3. Add title, x-label "Predicted", y-label "Actual".

    Args:
        cm:           2x2 confusion matrix from sklearn.
        class_names:  List of class label strings.
    """
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    raise NotImplementedError("TODO 13: implement plot_confusion_matrix()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 14 ──────────────────────────────────────────────────────────────────
def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, model_name: str = "Model") -> None:
    """
    Plot the ROC curve and annotate with AUC score.

    Tasks:
        1. Compute fpr, tpr, _ = roc_curve(y_true, y_probs).
        2. Compute auc_score = roc_auc_score(y_true, y_probs).
        3. Plot fpr vs tpr, add a diagonal dashed reference line.
        4. Label axes and include AUC in the legend.

    Args:
        y_true:     Ground-truth binary labels.
        y_probs:    Predicted probabilities (sigmoid output).
        model_name: Name shown in the plot legend.
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    raise NotImplementedError("TODO 14: implement plot_roc_curve()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 15 ──────────────────────────────────────────────────────────────────
def grad_cam(
    model: keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single image.

    Tasks:
        1. Build a sub-model that outputs (last_conv_layer output, model output).
           Use tf.keras.Model(inputs=model.inputs,
                              outputs=[model.get_layer(last_conv_layer_name).output,
                                       model.output]).
        2. Use tf.GradientTape to record gradients of the class score with
           respect to the last conv layer activations.
        3. Pool gradients spatially (mean over H and W axes).
        4. Weight each channel of the feature map by its pooled gradient.
        5. Apply ReLU and normalise to [0, 1].
        6. Return the heatmap as a 2-D numpy array.

    Args:
        model:               Trained keras.Model with conv layers.
        img_array:           Preprocessed image, shape (1, H, W, C).
        last_conv_layer_name: Name of the last convolutional layer.

    Returns:
        np.ndarray of shape (h, w), values in [0, 1].
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    raise NotImplementedError("TODO 15: implement grad_cam()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


def compare_models(results: dict) -> None:
    """
    Bar chart comparing accuracy, precision, recall, F1, and AUC
    across multiple models.

    Already implemented — do not modify.

    Args:
        results: {model_name: metrics_dict} where metrics_dict is the output
                 of evaluate_model().
    """
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.show()
