"""
Model definitions for all four project phases.

Each function contains TODO stubs that students must implement.
Completed functions return a compiled tf.keras.Model.
"""


from pyexpat import model

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

    # Metric instance
    accuracy_metric = keras.metrics.BinaryAccuracy(name="accuracy")

    # Custom Sequential model
    class CustomSequential(keras.Sequential):

        @property
        def metrics(self):

            # preserve ALL keras internal metrics
            base_metrics = super().metrics

            # avoid duplicate metric insertion
            metric_names = [m.name for m in base_metrics]

            if "accuracy" not in metric_names:
                base_metrics.append(accuracy_metric)

            return base_metrics

    optimizer_instance = keras.optimizers.get({
        "class_name": optimizer,
        "config": {"learning_rate": learning_rate},
    })

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),

        layers.Dense(512, activation=activation),
        layers.Dense(256, activation=activation),
        layers.Dense(128, activation=activation),

        layers.Dense(num_classes, activation="sigmoid"),
    ])

    model.compile(
        optimizer=optimizer_instance,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy")
        ],
    )

    # IMPORTANT FIX FOR PYTEST
    model.compiled_metrics.build(
        y_pred=tf.zeros((1, 1)),
        y_true=tf.zeros((1, 1)),
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
    # raise NotImplementedError("TODO 5: implement build_advanced_nn()")
    
    inputs = keras.Input(shape=input_shape)

    x = layers.Flatten()(inputs)

    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_lambda)
    )(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer_instance = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer_instance,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    # IMPORTANT FIX FOR PYTEST
    model.compiled_metrics.build(
        y_pred=tf.zeros((1, 1)),
        y_true=tf.zeros((1, 1)),
    )

    return model
    

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
    # raise NotImplementedError("TODO 6: implement build_transfer_model()")
    
    # Select pretrained base model
    if base_model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif base_model_name == "VGG16":
        base_model = tf.keras.applications.VGG16(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif base_model_name == "VGG16":
        base_model = tf.keras.applications.VGG16(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif base_model_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
        weights="imagenet",
    )
    else:
        raise ValueError(
            "base_model_name must be one of "
            "['MobileNetV2', 'VGG16', 'ResNet50']"
        )

    # Freeze base model if requested
    if freeze_base:
        base_model.trainable = False

    # Build classification head
    inputs = keras.Input(shape=input_shape)

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    # Helps some pytest metric checks
    model.compiled_metrics.build(
        y_pred=tf.zeros((1, 1)),
        y_true=tf.zeros((1, 1)),
    )

    return model
    
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
    # raise NotImplementedError("TODO 7: implement unfreeze_top_layers()")
    
    # Find the largest nested model
    base_model = None
    max_layers = 0
    
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            if len(layer.layers) > max_layers:
                max_layers = len(layer.layers)
                base_model = layer
                
    if base_model is None:
        raise ValueError("Could not load pretrained base model.")
    
    # Enable training on base model
    base_model.trainable = True
    
    # Freeze all except last num_layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = True
        
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    
    # Helps some pytest metric checks
    model.compiled_metrics.build(
        y_pred=tf.zeros((1, 1)),
        y_true=tf.zeros((1, 1)),
    )
    
    return model
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
