"""
CNN+GRU model for cardiac and pulmonary sound classification.
Architecture: mel-spectrogram → CNN encoder → GRU temporal → classifier
"""
from __future__ import annotations
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

LABELS = [
    "normal", "murmur", "aortic_stenosis",
    "mitral_stenosis", "crackles", "wheezes",
    "bronchial", "pleural_rub",
]
NUM_CLASSES = len(LABELS)


def build_cnn_gru(
    n_mels: int = 128,
    n_frames: int = 128,
    n_classes: int = NUM_CLASSES,
    cnn_filters: Tuple[int, ...] = (32, 64, 128),
    gru_units: int = 128,
    dropout: float = 0.3,
):
    """
    CNN+GRU hybrid for audio classification.

    Input shape: (batch, n_mels, n_frames, 1)  — mel spectrogram
    Returns: Keras Model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input
    except ImportError:
        raise ImportError("pip install tensorflow")

    inp = Input(shape=(n_mels, n_frames, 1), name="spectrogram")
    x = inp

    # CNN encoder — extracts local spectral patterns
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv2D(filters, (3, 3), padding="same", name=f"conv{i+1}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout / 2)(x)

    # Reshape for RNN: (batch, time_steps, features)
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # Bidirectional GRU — captures temporal dynamics
    x = layers.Bidirectional(
        layers.GRU(gru_units, return_sequences=True, dropout=dropout, recurrent_dropout=0.1),
        name="bigru_1"
    )(x)
    x = layers.Bidirectional(
        layers.GRU(gru_units // 2, dropout=dropout, recurrent_dropout=0.1),
        name="bigru_2"
    )(x)

    # Attention mechanism — weight important time steps
    # (simplified: global average + max pool concat)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="CNN_GRU_Auscultation")
    return model


def compile_model(model, learning_rate: float = 1e-3):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("pip install tensorflow")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model


def get_callbacks(checkpoint_path: str, patience: int = 10):
    try:
        from tensorflow.keras import callbacks
    except ImportError:
        return []

    return [
        callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
        callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
    ]
