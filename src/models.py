"""Keras models: custom CNN, MobileNetV2 (ImageNet), bidirectional GRU for phrases."""

import tensorflow as tf
from tensorflow.keras import layers, models


def augment_block():
    return tf.keras.Sequential([
        layers.RandomRotation(0.08),          # about +-15 degrees
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.25, value_range=(0, 255)),
        layers.RandomContrast(0.25),
        # NO horizontal flip. Flipping changes handedness and sign meaning.
    ], name="augment")


def build_cnn(num_classes, size=128):
    inp = layers.Input((size, size, 3))
    x = augment_block()(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="asl_cnn")


def build_mobilenet(num_classes, size=224):
    inp = layers.Input((size, size, 3))
    x = augment_block()(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)      # MobileNetV2 expects [-1, 1]
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(size, size, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="asl_mobilenet")


def unfreeze_top(model, n_layers=30):
    """Stage 2 fine-tuning: unfreeze last n_layers of the backbone, keep BatchNorm frozen."""
    base = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def build_sequence_model(num_classes, seq_len=30, n_feat=140):
    inp = layers.Input((seq_len, n_feat))
    x = layers.GaussianNoise(0.01)(inp)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="phrase_gru")
