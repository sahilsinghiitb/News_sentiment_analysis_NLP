# Sentiment Analysis of Financial News using Classical ML & Transformers

## Overview
This NLP project focuses on performing sentiment analysis on financial news articles (4000+ instances) to classify statements into **positive**, **negative**, or **neutral** categories. The objective is to compare traditional machine learning approaches with transformer-based fine-tuning while handling class imbalance and extracting meaningful language-based features.

---

## Dataset Details
- **Total Records:** ~4000 financial news sentences
- **Sentiment Distribution:**
  - Neutral: ~60%
  - Positive: ~32%
  - Negative: ~8%
- Significant class imbalance addressed using **SMOTE**.

---

## Key Steps & Techniques

### ✅ Exploratory Data Analysis & Preprocessing
- Tokenization, stemming, stopword & punctuation removal
- Feature extraction: number of words, characters, sentences
- Used **TextBlob** for initial polarity scoring
- Detected skewness and neutral dominance in polarity distribution

### ✅ Feature Engineering & Vectorization
- Bag of Words (BoW), **TF-IDF**, and CBOW embeddings
- Similarity metrics: cosine, Manhattan, Euclidean
- N-Gram analysis (uni-gram, bi-gram), word cloud visualization

### ✅ Handling Class Imbalance
- Applied **SMOTE** to balance minority classes
- Compared model performance before and after resampling

---

## Model Training & Evaluation

| Model / Method (TF-IDF)               | Metric (F1) |
|--------------------------------------|-------------|
| Cosine Similarity + LogisticRegression | 0.73 (SMOTE) |
| RandomForest (Euclidean / Manhattan) | ~0.71 |
| **Hugging Face Transformer (Fine-Tuned)** | **0.889 Accuracy** |

- Fine-tuned a pre-trained transformer model (ProsusAI/finbert)
- Achieved **88.9% accuracy** — best performing method overall

---

## Conclusion
- Traditional ML models performed moderately well with TF-IDF + LR and similarity metrics
- SMOTE significantly improved recall for underrepresented classes
- **Transformer-based fine-tuning provided the highest accuracy (88.9%)**, demonstrating strong capability in finance domain sentiment tasks

---

## Tools & Libraries
- Python, Pandas, Numpy, Matplotlib
- NLTK, TextBlob, Scikit-learn
- imblearn (SMOTE)
- Hugging Face Transformers Pipeline

---

## Future Work
- Deploy the model using Flask or Streamlit for real-time sentiment tagging
- Integrate Named Entity Recognition (NER) to extract company names and financial indicators
