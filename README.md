# Decoding the Rhythms of Emotion

Text mining project on genre classification from song lyrics and sentiment analysis across genres.

## Project files

- `dataset/`: training data, unlabeled test data, and the sample submission template.
- `notebooks/EDA.ipynb`: data quality, class balance, lyric structure, and descriptive exploration.
- `notebooks/Preprocessing.ipynb`: cleaning choices and preprocessing comparisons; writes the intermediate pickle used below.
- `notebooks/Genre_Classification.ipynb`: model selection, assessment, error analysis, and test predictions.
- `notebooks/Sentiment_Analysis.ipynb`: sentiment scoring and comparisons by genre, time, and popularity.
- `utils/`: shared cleaning and sentiment functions.
- `submissions/submission.csv`: generated genre predictions aligned to the sample submission IDs.

## Run order

The train and test CSVs use Git LFS. After cloning, run `git lfs install` and `git lfs pull` before opening the notebooks.

Run notebooks from the `notebooks/` directory in this order:

1. `EDA.ipynb`
2. `Preprocessing.ipynb` (creates `../tmp/preprocessed_data.pickle`)
3. `Genre_Classification.ipynb`
4. `Sentiment_Analysis.ipynb`

The intermediate `tmp/` folder is ignored by Git; regenerate it by running preprocessing before classification. The notebooks use pandas, NumPy, scikit-learn, NLTK, Matplotlib, and Seaborn.

## Modeling notes

The classifier combines TF-IDF with artist, featured-artist, section, lyric-structure, release-year, and views features. Artist-prior and same-artist score adjustments are selected on an inner split; the outer holdout is used for assessment. Sentence embeddings are optional and off by default.
