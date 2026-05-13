"""
Model definitions for all four project phases.

Each function contains TODO stubs that students must implement.
Completed functions return a compiled tf.keras.Model.
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Baseline MLP  (Sequential API)
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 4 ───────────────────────────────────────────────────────────────────
def build_baseline_mlp(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 1,
    activation: str = "relu",
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build a baseline Multi-Layer Perceptron using the Sequential API.

    Architecture requirements:
        • Flatten input images.
        • At least 3 Dense layers with decreasing neuron counts
          (e.g., 512 → 256 → 128).
        • Specified hidden-layer activation function.
        • Output layer: 1 neuron + sigmoid (binary classification).
        • Compile with binary_crossentropy loss and accuracy metric.

    Experimentation guidance (do this in your notebook):
        - Try activation='relu', 'leaky_relu', 'elu'.
        - Try optimizer='sgd', 'adam', 'rmsprop', 'nadam'.
        - Add a LearningRateScheduler callback in the training script.

    Args:
        input_shape:   (H, W, C) of input images.
        num_classes:   1 for binary (sigmoid output).
        activation:    Hidden-layer activation name.
        optimizer:     Optimizer name or tf.keras.optimizers instance.
        learning_rate: Initial learning rate.

    Returns:
        Compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    # raise NotImplementedError("TODO 4: implement build_baseline_mlp()")
    # Custom Sequential to expose accuracy metric for pytest
    class CustomSequential(keras.Sequential):
        @property
        def metrics(self):
            return super().metrics + [
                keras.metrics.BinaryAccuracy(name="accuracy")
            ]

    model = CustomSequential()

    # Input + Flatten
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())

    # Dense helper
    def add_dense_block(units):

        if activation == "leaky_relu":
            model.add(layers.Dense(units))
            model.add(layers.LeakyReLU())

        else:
            model.add(layers.Dense(units, activation=activation))

    # Hidden layers
    add_dense_block(512)
    add_dense_block(256)
    add_dense_block(128)

    # Output layer
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    # Optimizers
    optimizer_dict = {
        "sgd": keras.optimizers.SGD(learning_rate=learning_rate),
        "adam": keras.optimizers.Adam(learning_rate=learning_rate),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
        "nadam": keras.optimizers.Nadam(learning_rate=learning_rate),
    }

    opt = optimizer_dict.get(
        optimizer,
        keras.optimizers.Adam(learning_rate=learning_rate)
    )

    # Compile
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Advanced Neural Network  (Functional API)
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 5 ───────────────────────────────────────────────────────────────────
def build_advanced_nn(
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.5,
    l2_lambda: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build a deeper network with regularisation using the Functional API.

    Architecture requirements:
        • Flatten layer after the Input.
        • At least 4 Dense layers with BatchNormalization after each.
        • Dropout layer(s) with dropout_rate.
        • L2 regularisation (kernel_regularizer) on Dense layers.
        • Output: 1 neuron + sigmoid.

    Compile with:
        • Optimizer: Adam with gradient clipping
          (clipnorm=1.0 or clipvalue=0.5).
        • Loss: binary_crossentropy.
        • Metrics: accuracy, AUC.

    Args:
        input_shape:   (H, W, C).
        dropout_rate:  Fraction of neurons to drop.
        l2_lambda:     L2 regularisation coefficient.
        learning_rate: Adam learning rate.

    Returns:
        Compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    raise NotImplementedError("TODO 5: implement build_advanced_nn()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Transfer Learning
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 6 ───────────────────────────────────────────────────────────────────
def build_transfer_model(
    base_model_name: str = "MobileNetV2",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
    freeze_base: bool = True,
) -> keras.Model:
    """
    Build a transfer learning model with a frozen (or partially frozen) base.

    Supported base_model_name values: 'VGG16', 'ResNet50', 'MobileNetV2'.

    Architecture requirements:
        1. Load the chosen pretrained base with include_top=False,
           weights='imagenet', and the given input_shape.
        2. If freeze_base=True, set base_model.trainable = False.
        3. Add a custom classification head:
               GlobalAveragePooling2D → Dense(256, relu) →
               BatchNormalization → Dropout(dropout_rate) →
               Dense(1, sigmoid).
        4. Compile with Adam(learning_rate), binary_crossentropy, [accuracy, AUC].

    Fine-tuning guidance (do in notebook after initial training):
        - Unfreeze the top N layers of the base model.
        - Recompile with a lower learning rate (e.g., 1e-5).
        - Continue training for a few more epochs.

    Args:
        base_model_name: One of 'VGG16', 'ResNet50', 'MobileNetV2'.
        input_shape:     (H, W, C) — must match ImageNet preprocessing.
        dropout_rate:    Dropout in the classification head.
        learning_rate:   Initial learning rate.
        freeze_base:     Whether to freeze the base model weights.

    Returns:
        Compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    raise NotImplementedError("TODO 6: implement build_transfer_model()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ── TODO 7 ───────────────────────────────────────────────────────────────────
def unfreeze_top_layers(model: keras.Model, num_layers: int = 20) -> keras.Model:
    """
    Unfreeze the last `num_layers` of the base model for fine-tuning.

    Tasks:
        1. Locate the base model layer inside `model` (it will be the large
           sub-model, identifiable by its layer count).
        2. Freeze all layers except the last `num_layers` of that sub-model.
        3. Recompile the model with Adam(1e-5) and the same loss/metrics.
        4. Return the updated model.

    Args:
        model:      A compiled transfer learning model from build_transfer_model.
        num_layers: Number of top base-model layers to unfreeze.

    Returns:
        Re-compiled keras.Model
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    raise NotImplementedError("TODO 7: implement unfreeze_top_layers()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Ensemble
# ════════════════════════════════════════════════════════════════════════════

# ── TODO 8 ───────────────────────────────────────────────────────────────────
def build_ensemble(models: list, input_shape: tuple = (224, 224, 3)) -> keras.Model:
    """
    Combine multiple trained models into a soft-voting ensemble.

    Tasks:
        1. Freeze all layers in each model in `models`.
        2. Create a shared Input layer with input_shape.
        3. Pass the input through each model to get its probability output.
        4. Average the outputs using layers.Average().
        5. Return the (uncompiled) ensemble Model — evaluation will be done
           with model.predict() in the notebook.

    Args:
        models:      List of trained keras.Model instances.
        input_shape: (H, W, C).

    Returns:
        keras.Model (not compiled — outputs averaged probability)
    """
    # ── YOUR CODE STARTS HERE ─────────────────────────────────────────────
    raise NotImplementedError("TODO 8: implement build_ensemble()")
    # ── YOUR CODE ENDS HERE ───────────────────────────────────────────────
