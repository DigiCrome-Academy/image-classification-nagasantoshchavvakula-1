"""
Training utilities — callbacks, schedules, and the main training loop.
Students implement the TODO sections; the rest is scaffolding.
"""

import math
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


CHECKPOINT_DIR = Path("models")


# ── TODO 9 ───────────────────────────────────────────────────────────────────
def get_callbacks(
    model_name: str = "model",
    checkpoint_dir: Path = CHECKPOINT_DIR,
    patience: int = 5,
    monitor: str = "val_loss",
) -> list:
    """
    Build and return a list of Keras callbacks for training.

    Required callbacks:
        1. ModelCheckpoint — save the best model to
           checkpoint_dir / f"{model_name}_best.keras"
           with save_best_only=True, monitor=monitor.
        2. EarlyStopping — stop training when monitor does not improve
           for `patience` epochs; restore_best_weights=True.
        3. ReduceLROnPlateau — reduce LR by factor=0.5 when monitor
           plateaus for patience//2 epochs; min_lr=1e-7.
        4. TensorBoard — log to logs/{model_name}/.

    Args:
        model_name:     Base name used for file paths.
        checkpoint_dir: Directory to save checkpoints.
        patience:       Early-stopping patience (epochs).
        monitor:        Metric to monitor ('val_loss' or 'val_accuracy').

    Returns:
        List of keras.callbacks.Callback
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 9: implement get_callbacks()")
        # Checkpoint file path
    checkpoint_path = checkpoint_dir / f"{model_name}_best.keras"

    callbacks = [

        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),

        # Stop training early if validation metric stops improving
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),

        # Reduce learning rate when validation metric plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=f"logs/{model_name}"
        ),
    ]

    return callbacks
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 10 ──────────────────────────────────────────────────────────────────
def get_lr_schedule(
    schedule_type: str = "cosine",
    initial_lr: float = 1e-3,
    epochs: int = 30,
) -> keras.callbacks.LearningRateScheduler:
    """
    Return a LearningRateScheduler callback.

    Implement at least TWO of the following schedules selectable via
    schedule_type:
        'step'    — halve LR every 10 epochs.
        'cosine'  — cosine annealing from initial_lr to 0.
        'warmup'  — linear warm-up for 5 epochs then cosine decay.

    Args:
        schedule_type: One of 'step', 'cosine', 'warmup'.
        initial_lr:    Starting learning rate.
        epochs:        Total training epochs (used by cosine schedule).

    Returns:
        keras.callbacks.LearningRateScheduler
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 10: implement get_lr_schedule()")
    def lr_schedule(epoch, lr):
        
        # Step Decay
        if schedule_type == "step":
            return initial_lr * (0.5 ** (epoch // 10))
        
        # Cosine Annealing
        elif schedule_type == "cosine":
            cosine_decay = 0.5 * (
                1+math.cos(math.pi * epoch / epochs)
            )
            return initial_lr * cosine_decay
        
        # Warm-up + Cosine Decay
        elif schedule_type == "warmup":
            
            warmup_epochs = 5
            # linear warm-up
            if epoch < warmup_epochs:
                return initial_lr * ((epoch + 1) / warmup_epochs)
            
            # cosine decay after warm-up
            decay_epochs = epochs - warmup_epochs
            cosine_decay = 0.5 * (
                1 + math.cos(math.pi * (epoch - warmup_epochs) / decay_epochs)
            )
            return initial_lr * cosine_decay
        # Default to initial_lr if schedule_type is unrecognized
        return initial_lr
    return keras.callbacks.LearningRateScheduler(lr_schedule)
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 11 ──────────────────────────────────────────────────────────────────
def train_model(
    model: keras.Model,
    train_data,
    val_data,
    epochs: int = 30,
    class_weights: dict = None,
    callbacks: list = None,
) -> keras.callbacks.History:
    """
    Train `model` and return the History object.

    Tasks:
        1. Call model.fit() with train_data, validation_data=val_data,
           epochs=epochs, class_weight=class_weights, callbacks=callbacks.
        2. Return the History object for later plotting.

    Args:
        model:         Compiled keras.Model.
        train_data:    Training generator or tf.data.Dataset.
        val_data:      Validation generator or tf.data.Dataset.
        epochs:        Number of training epochs.
        class_weights: Optional dict for imbalanced datasets.
        callbacks:     List of Keras callbacks.

    Returns:
        keras.callbacks.History
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 11: implement train_model()")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


def plot_history(history: keras.callbacks.History, title: str = "Training") -> None:
    """
    Plot accuracy and loss curves side by side.
    Already implemented — do not modify.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, val_metric in zip(
        axes,
        ["accuracy", "loss"],
        ["val_accuracy", "val_loss"],
    ):
        ax.plot(history.history.get(metric, []), label="Train")
        ax.plot(history.history.get(val_metric, []), label="Val")
        ax.set_title(f"{title} — {metric}")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.show()
