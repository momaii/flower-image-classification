"""
Model definitions for flower image classification.

Two architectures are provided:
- build_cnn_v1 / build_cnn_v2 : custom CNNs trained from scratch
- build_densenet             : DenseNet121 transfer learning model
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input


# ---------------------------------------------------------------------------
# Shared data augmentation pipelines
# ---------------------------------------------------------------------------

def get_augmentation_v1() -> tf.keras.Sequential:
    """Light augmentation used in CNN v1."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation_v1")


def get_augmentation_v2() -> tf.keras.Sequential:
    """Stronger augmentation used in CNN v2."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ], name="augmentation_v2")


# ---------------------------------------------------------------------------
# CNN trained from scratch — v1 (VGG-style, ~1.28 M params)
# ---------------------------------------------------------------------------

def build_cnn_v1(num_classes: int = 4) -> models.Sequential:
    """
    VGG-style CNN with 4 convolutional blocks (64 → 128 → 256 → 256 filters).

    Blocks 1 and 2 use two consecutive convolutions before pooling to increase
    receptive field depth. Blocks 3 and 4 use a single convolution.
    Global Average Pooling replaces Flatten to reduce parameter count and
    limit overfitting in a low-data regime.

    ~1.28 M trainable parameters.
    """
    return models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1.0 / 255),
        get_augmentation_v1(),

        # Block 1 — 64 filters × 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Block 2 — 128 filters × 2
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Block 3 — 256 filters × 1
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Block 4 — 256 filters × 1
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_v1")


# ---------------------------------------------------------------------------
# CNN trained from scratch — v2 (reduced capacity, stronger regularisation)
# ---------------------------------------------------------------------------

def build_cnn_v2(
    num_classes: int = 4,
    learning_rate: float = 1e-4,
) -> models.Sequential:
    """
    Revised CNN with reduced capacity and stronger regularisation to reduce
    overfitting in a low-data regime.

    Each block uses a single convolution. The dense head is halved (256 units)
    and a lower learning rate is used. ReduceLROnPlateau is recommended during
    training.

    ~1.03 M trainable parameters.
    """
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1.0 / 255),
        get_augmentation_v2(),

        # Block 1
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_v2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Transfer learning — DenseNet121 (frozen backbone)
# ---------------------------------------------------------------------------

def build_densenet(num_classes: int = 4) -> Model:
    """
    DenseNet121 feature extractor with a lightweight classification head.

    The backbone is loaded with ImageNet weights and kept fully frozen.
    Only the Dense + Dropout head is trained, limiting the number of
    trainable parameters to ~263 K while benefiting from rich pretrained
    visual representations.

    Args:
        num_classes: Number of output classes (default 4).

    Returns:
        Compiled Keras Model.
    """
    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3),
    )
    base_model.trainable = False

    inputs = Input(shape=(128, 128, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="densenet121_transfer")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
