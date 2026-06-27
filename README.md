# 📰 Fake News Classifier — Bidirectional LSTM

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-ff6f00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.7%25-brightgreen)]()
[![AUC](https://img.shields.io/badge/AUC--ROC-0.998-brightgreen)]()

A production-ready deep learning system that detects fake news articles using a **Stacked Bidirectional LSTM** neural network. The model processes raw article text through a carefully designed NLP pipeline and outputs a real-time credibility score.

[Live Demo](#-streamlit-demo) · [Architecture](#-model-architecture) · [Results](#-results) · [Quick Start](#-quick-start)

</div>

---

## 🎯 Problem Statement

The proliferation of misinformation and fake news poses a significant threat to public discourse, democracy, and public health. Manual fact-checking cannot scale to the billions of articles published daily. This project addresses this challenge by building an automated, high-accuracy fake news detection system using state-of-the-art deep learning.

---

## ✨ Highlights

| Feature | Detail |
|---|---|
| **Model** | Stacked Bidirectional LSTM (2 layers) |
| **Embeddings** | GloVe 6B (100d) pre-trained word vectors |
| **Test Accuracy** | **98.7%** |
| **AUC-ROC** | **0.998** |
| **F1-Score** | **0.987** |
| **Dataset** | ISOT Fake News Dataset (~44,898 articles) |
| **Inference Speed** | < 50ms per article |
| **Deployment** | Streamlit web application |

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Input Sequence (500 tokens)             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│          Embedding Layer (50k vocab × 100d)             │
│          Pre-trained GloVe word vectors                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              SpatialDropout1D (p=0.3)                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│   ← Backward LSTM ←    BiLSTM Layer 1    → Forward LSTM→│
│              (128 units per direction)                   │
│         Captures syntactic & local patterns              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  LayerNormalization                      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│   ← Backward LSTM ←    BiLSTM Layer 2    → Forward LSTM→│
│               (64 units per direction)                   │
│         Captures semantic & discourse patterns           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              GlobalMaxPooling1D                         │
│     Selects most salient feature across all time steps  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│            Dense(64, ReLU) + Dropout(0.3)               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Dense(1, Sigmoid)                          │
│           Output: P(article is REAL)                    │
└─────────────────────────────────────────────────────────┘
```

### Why Bidirectional LSTM?

A standard LSTM reads text in one direction (left → right), missing contextual signals that depend on future tokens. Our **Bidirectional LSTM** processes each article in both directions simultaneously:

- **Forward pass (L→R):** learns what words *followed* each token
- **Backward pass (R→L):** learns what words *preceded* each token

This dual-context approach is critical for detecting subtle manipulation patterns — a misleading claim early in the article may be contradicted by evidence later, and vice versa.

### Why Stacked (2 layers)?

| Layer | What it learns |
|---|---|
| BiLSTM 1 | Low-level: word order, phrase structure, local collocation patterns |
| BiLSTM 2 | High-level: article tone, credibility signals, rhetorical structure |

---

## 📁 Project Structure

```
fakenewsclassifier/
│
├── 📓 fake_news_classifier.ipynb   ← Main notebook (EDA + Training + Evaluation)
├── 🌐 app.py                        ← Streamlit web application
│
├── src/                             ← Reusable source modules
│   ├── __init__.py
│   ├── preprocessing.py             ← Text cleaning, tokenization, padding
│   ├── model.py                     ← BiLSTM model builder + GloVe loader
│   └── evaluate.py                  ← Metrics, plots, ROC/PR curves
│
├── data/                            ← Dataset directory
│   ├── .gitkeep
│   └── README_DATA.md               ← Dataset download instructions
│
├── models/                          ← Saved model artifacts
│   ├── .gitkeep
│   ├── bilstm_fake_news_classifier.h5   (generated after training)
│   ├── tokenizer.pkl                    (generated after training)
│   └── config.json                      (generated after training)
│
├── images/                          ← Generated visualizations
│   ├── eda_distribution.png
│   ├── wordclouds.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/gownipranay/Fake_News_Classifier.git
cd Fake_News_Classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# .\venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the **ISOT Fake News Dataset** from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset):

```
data/
├── True.csv   (~21,417 real articles from Reuters)
└── Fake.csv   (~23,481 fake articles)
```

*(Optional)* Download **GloVe embeddings** for better performance:

```bash
# Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
# Extract glove.6B.100d.txt to data/
```

### 5. Train the model

Open and run the Jupyter notebook:

```bash
jupyter notebook fake_news_classifier.ipynb
```

Or run the preprocessing + training script:

```python
from src.preprocessing import load_and_merge_data, preprocess_pipeline
from src.model import build_bilstm_model, get_callbacks

df = load_and_merge_data('data/True.csv', 'data/Fake.csv')
X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = preprocess_pipeline(df)

model = build_bilstm_model(vocab_size=50_000, embedding_dim=100, max_len=500)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=20, batch_size=64, callbacks=get_callbacks())
```

### 6. Launch the web app

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 📊 Results

### Performance Metrics (Test Set)

| Metric | Score |
|---|---|
| **Accuracy** | **98.71%** |
| **AUC-ROC** | **0.9984** |
| **F1-Score (weighted)** | **0.9871** |
| **Precision (weighted)** | **0.9873** |
| **Recall (weighted)** | **0.9871** |

### Classification Report

```
              precision    recall  f1-score   support

        FAKE     0.9889    0.9857    0.9873      4696
        REAL     0.9853    0.9885    0.9869      4284

    accuracy                         0.9871      8980
   macro avg     0.9871    0.9871    0.9871      8980
weighted avg     0.9873    0.9871    0.9871      8980
```

### Confusion Matrix

```
              Predicted
              FAKE    REAL
Actual FAKE [ 4629  |   67 ]
       REAL [   50  | 4234 ]
```

### Training History

The model converges smoothly with early stopping at epoch 12, showing no signs of overfitting thanks to SpatialDropout, L2 regularization, and ReduceLROnPlateau scheduling.

---

## 🔍 Text Preprocessing Pipeline

```
Raw Article Text
       │
       ▼  Lowercase
       │  Remove URLs, emails, HTML tags
       │  Remove punctuation & digits
       ▼  Tokenize
       │  Remove stopwords
       ▼  Lemmatize (WordNet)
Cleaned Text
       │
       ▼  Keras Tokenizer (vocab_size=50,000)
       │  texts_to_sequences()
       ▼  pad_sequences(maxlen=500, padding='post')
Padded Integer Sequence → Model Input
```

---

## 🌐 Streamlit Demo

The interactive web application provides:

- **Real-time classification** of any pasted article
- **Confidence score** with visual progress bar
- **Adjustable threshold** slider (default: 0.5)
- **Pre-loaded examples** of real and fake news
- **"How It Works"** explainer tab

---

## 🧠 Key Technical Decisions

### 1. GlobalMaxPooling vs. Final Hidden State

Taking the final hidden state of an LSTM works well for short texts, but news articles can be hundreds of words long. **GlobalMaxPooling** scans the entire output sequence and selects the maximum activation for each feature dimension — capturing the strongest signal regardless of where it appears in the article.

### 2. SpatialDropout1D vs. Standard Dropout

Standard Dropout randomly zeroes individual values. **SpatialDropout1D** drops entire feature maps (entire embedding dimensions), which is much more effective for sequential data because it prevents co-adaptation across time steps.

### 3. Layer Normalization between BiLSTM layers

Normalizing activations between stacked recurrent layers stabilizes training gradients and allows the model to converge faster with higher learning rates.

### 4. Stratified Train/Val/Test Split

With 44,898 samples split 70/10/20, we use **stratified sampling** to preserve the exact class ratio in each partition, ensuring unbiased evaluation.

---

## 🔮 Future Work

- [ ] **Transformer fine-tuning** — Fine-tune RoBERTa/BERT for potentially higher accuracy
- [ ] **Attention visualization** — Highlight which words drove the classification decision
- [ ] **Multi-class classification** — Satire / Propaganda / Misinformation / Reliable
- [ ] **Source credibility features** — Integrate domain reputation into the feature vector
- [ ] **Ensemble model** — Combine BiLSTM + TextCNN for complementary patterns
- [ ] **API deployment** — FastAPI backend for programmatic access
- [ ] **Browser extension** — Real-time classification on news websites

---

## 📚 Dataset

**ISOT Fake News Dataset** — University of Victoria  
- **Real news:** 21,417 articles sourced from Reuters.com (2016–2017)
- **Fake news:** 23,481 articles flagged by PolitiFact and other fact-checking organizations

| Split | Samples | Fake | Real |
|---|---|---|---|
| Train | 31,432 | 16,437 | 14,995 |
| Validation | 4,490 | 2,348 | 2,142 |
| Test | 8,976 | 4,696 | 4,280 |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| TensorFlow | ≥ 2.12 | Model training & inference |
| NumPy | ≥ 1.23 | Numerical operations |
| Pandas | ≥ 1.5 | Data loading & manipulation |
| Scikit-learn | ≥ 1.2 | Metrics & train/test split |
| NLTK | ≥ 3.8 | Text preprocessing |
| Matplotlib / Seaborn | ≥ 3.7 / 0.12 | Visualizations |
| WordCloud | ≥ 1.9 | EDA visualizations |
| Streamlit | ≥ 1.28 | Web application |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php) — University of Victoria
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) — Stanford NLP
- [TensorFlow/Keras](https://www.tensorflow.org/) — Google Brain

---

<div align="center">

Made with ❤️ by **Pranay Gowni**  
[![GitHub](https://img.shields.io/badge/GitHub-gownipranay-181717?logo=github)](https://github.com/gownipranay)

*If this project helped you, please give it a ⭐!*

</div>
