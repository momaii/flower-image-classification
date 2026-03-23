import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input


# Augmentation légère pour CNN v1
def get_augmentation_v1():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation_v1")


# Augmentation plus forte pour CNN v2 (lutte contre le surapprentissage)
def get_augmentation_v2():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ], name="augmentation_v2")


def build_cnn_v1(num_classes=4):
    """CNN from scratch, style VGG, ~1.28M paramètres.
    4 blocs convolutionnels (64 -> 128 -> 256 -> 256 filtres).
    Blocs 1 et 2 : deux convolutions successives avant pooling.
    Global Average Pooling à la place de Flatten pour réduire le surapprentissage.
    """
    return models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1.0 / 255),
        get_augmentation_v1(),

        # Bloc 1 — 64 filtres x2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Bloc 2 — 128 filtres x2
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Bloc 3 — 256 filtres
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Bloc 4 — 256 filtres
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_v1")


def build_cnn_v2(num_classes=4, learning_rate=1e-4):
    """CNN v2 : capacité réduite + régularisation renforcée par rapport à v1.
    Un seul bloc convolutionnel par niveau, tête Dense réduite (256 unités),
    learning rate plus faible. Meilleure généralisation sur données limitées.
    ~1.03M paramètres.
    """
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1.0 / 255),
        get_augmentation_v2(),

        # Bloc 1
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Bloc 2
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Bloc 3
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        # Bloc 4
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


def build_densenet(num_classes=4):
    """Transfert d'apprentissage avec DenseNet121 pré-entraîné sur ImageNet.
    Backbone gelé (feature extractor uniquement), tête de classification légère.
    Seule la tête est entraînée (~263K paramètres entraînables).
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
