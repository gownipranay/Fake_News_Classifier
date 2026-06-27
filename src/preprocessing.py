"""
Text preprocessing pipeline for Fake News Classification.
Handles cleaning, tokenization, and sequence padding.
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


download_nltk_resources()

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Full text cleaning pipeline.

    Steps:
        1. Lowercase
        2. Remove URLs, emails, HTML tags
        3. Remove punctuation & digits
        4. Tokenize
        5. Remove stopwords (optional)
        6. Lemmatize (optional)
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def build_tokenizer(texts: List[str], vocab_size: int = 50_000) -> Tokenizer:
    """Fit a Keras Tokenizer on the provided texts."""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_sequences(
    tokenizer: Tokenizer,
    texts: List[str],
    max_len: int = 500,
    padding: str = 'post',
    truncating: str = 'post',
) -> np.ndarray:
    """Convert cleaned texts to padded integer sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating)


def load_and_merge_data(true_path: str, fake_path: str) -> pd.DataFrame:
    """
    Load the WELFake / ISOT dataset CSVs and merge them.

    Expected columns: title, text (or text only).
    Labels: 1 = Real, 0 = Fake.
    """
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true['label'] = 1
    df_fake['label'] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Combine title + text if both exist
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['content'] = df['text'].fillna('')
    else:
        raise ValueError("Dataset must contain a 'text' column.")

    df = df[['content', 'label']].dropna().reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def preprocess_pipeline(
    df: pd.DataFrame,
    vocab_size: int = 50_000,
    max_len: int = 500,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple:
    """
    End-to-end preprocessing: clean → tokenize → split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, tokenizer
    """
    from sklearn.model_selection import train_test_split

    print("Cleaning text...")
    df['clean_content'] = df['content'].apply(clean_text)

    texts = df['clean_content'].tolist()
    labels = df['label'].values

    # Stratified split
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=42
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("Building tokenizer...")
    tokenizer = build_tokenizer(X_train, vocab_size=vocab_size)

    print("Converting to sequences...")
    X_train_seq = texts_to_sequences(tokenizer, X_train, max_len)
    X_val_seq   = texts_to_sequences(tokenizer, X_val,   max_len)
    X_test_seq  = texts_to_sequences(tokenizer, X_test,  max_len)

    return X_train_seq, X_val_seq, X_test_seq, y_train, y_val, y_test, tokenizer
