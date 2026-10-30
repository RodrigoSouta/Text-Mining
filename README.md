# Decoding the Rhythms of Emotion: Lyrics Classification & Sentiment Analysis

A Text Mining project focused on predicting musical genres from song lyrics and analyzing emotional undertones across musical genres.

---

## Project Overview

1. **Genre Identification:** Predicting one of six musical genres (`pop`, `rap`, `rock`, `rb`, `misc`, `country`) using NLP and text classification techniques on song lyrics.
2. **Sentiment Analysis:** Exploring emotional patterns, lexical polarity, and sentiment variations across genres, release eras, and popularity levels.

---

## Project Structure

```text
├── dataset/
│   ├── train.csv              
│   ├── test.csv               
│   └── sample_submission.csv 
├── notebooks/
│   ├── EDA.ipynb              
│   └── Preprocessing.ipynb   
├── utils/
│   ├── __init__.py
│   └── pipeline.py             
└── README.md
```

---

## Workflow

1. **Exploratory Data Analysis (`notebooks/EDA.ipynb`)**
   - Audits missing values, duplicates, and release year anomalies.
   - Evaluates genre imbalance (~70% Pop and Rap combined).
   - Examines lyric lengths, structural markers (`[Chorus]`, `[Verse]`), and term frequencies per genre.

2. **Preprocessing & Feature Engineering (`notebooks/Preprocessing.ipynb`)**
   - Applies dual cleaning strategies:
     - **Classification:** Stopwords and punctuation removed, verbs lemmatized, diacritics normalized.
     - **Sentiment Analysis:** Punctuation, capitalization, and stopwords preserved to maintain emotional valence and negation.
   - Compares Bag-of-Words, TF-IDF (unigrams + bigrams with sublinear scaling), and LSA dimensionality reduction.
   - Saves cleaned text and fitted vectorizers to `tmp/preprocessed_data.pickle` for modeling.
