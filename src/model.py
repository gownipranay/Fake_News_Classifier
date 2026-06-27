"""
Bidirectional LSTM model for Fake News Classification.

Architecture:
    Embedding → SpatialDropout1D → BiLSTM × 2 → GlobalMaxPool → Dense → Output
"""

import numpy as np
from typing import Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense,
    Dropout, SpatialDropout1D, GlobalMaxPooling1D,
    LayerNormalization, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


# ─────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────

def build_embedding_matrix(
    tokenizer,
    glove_path: str,
    embedding_dim: int = 100,
    vocab_size: Optional[int] = None,
) -> np.ndarray:
    """
    Build a GloVe embedding matrix aligned to the tokenizer's word index.

    Args:
        tokenizer: Fitted Keras Tokenizer.
        glove_path: Path to GloVe .txt file (e.g. glove.6B.100d.txt).
        embedding_dim: Dimensionality of GloVe vectors.
        vocab_size: Cap on vocabulary size; defaults to tokenizer.num_words.

    Returns:
        Numpy array of shape (vocab_size + 1, embedding_dim).
    """
    vocab_size = vocab_size or tokenizer.num_words or len(tokenizer.word_index)

    embeddings_index: Dict[str, np.ndarray] = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec

    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
    hits, misses = 0, 0
    for word, idx in tokenizer.word_index.items():
        if idx > vocab_size:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec
            hits += 1
        else:
            misses += 1

    print(f"GloVe coverage — hits: {hits:,} | misses: {misses:,}")
    return embedding_matrix


# ─────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────

def build_bilstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    max_len: int = 500,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.1,
    dense_units: int = 64,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
    embedding_matrix: Optional[np.ndarray] = None,
    trainable_embeddings: bool = True,
) -> Model:
    """
    Stacked Bidirectional LSTM classifier.

    Architecture overview:
        Input (max_len,)
        → Embedding (vocab_size+1, embedding_dim)   [optionally pre-trained]
        → SpatialDropout1D
        → BiLSTM(lstm_units, return_sequences=True)
        → LayerNormalization
        → BiLSTM(lstm_units // 2, return_sequences=True)
        → GlobalMaxPooling1D
        → Dense(dense_units, relu) + Dropout
        → Dense(1, sigmoid)

    Returns:
        Compiled Keras model.
    """
    inputs = Input(shape=(max_len,), name='input_sequence')

    # Embedding
    if embedding_matrix is not None:
        x = Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=trainable_embeddings,
            name='embedding',
        )(inputs)
    else:
        x = Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            input_length=max_len,
            embeddings_regularizer=l2(l2_reg),
            name='embedding',
        )(inputs)

    x = SpatialDropout1D(dropout_rate, name='spatial_dropout')(x)

    # First BiLSTM layer — returns full sequences
    x = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
        ),
        name='bilstm_1',
    )(x)
    x = LayerNormalization(name='layer_norm_1')(x)

    # Second BiLSTM layer — returns full sequences for pooling
    x = Bidirectional(
        LSTM(
            lstm_units // 2,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
        ),
        name='bilstm_2',
    )(x)

    # Global pooling captures the most salient features across all time steps
    x = GlobalMaxPooling1D(name='global_max_pool')(x)

    # Classifier head
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg), name='dense_1')(x)
    x = Dropout(dropout_rate, name='dropout')(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='BiLSTM_FakeNewsClassifier')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    return model


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────

def get_callbacks(
    checkpoint_path: str = 'models/best_model.h5',
    patience: int = 5,
    log_dir: str = 'logs/fit',
) -> list:
    """Standard callback suite."""
    return [
        EarlyStopping(
            monitor='val_auc',
            patience=patience,
            restore_best_weights=True,
            mode='max',
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]


def predict_single(model: Model, tokenizer, text: str, max_len: int = 500) -> Dict[str, Any]:
    """
    Classify a single article.

    Returns dict with keys: label, confidence, is_fake.
    """
    from src.preprocessing import clean_text, texts_to_sequences

    cleaned = clean_text(text)
    seq = texts_to_sequences(tokenizer, [cleaned], max_len)
    prob = float(model.predict(seq, verbose=0)[0][0])

    return {
        'label': 'REAL' if prob >= 0.5 else 'FAKE',
        'confidence': prob if prob >= 0.5 else 1 - prob,
        'raw_probability': prob,
        'is_fake': prob < 0.5,
    }
