import re
import os
import functools
import unicodedata
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from tqdm import tqdm

try:
    from unidecode import unidecode as _external_unidecode
    def unidecode(text):
        s = str(text)
        return s if s.isascii() else _external_unidecode(s)
except ImportError:
    def unidecode(text):
        s = str(text)
        if s.isascii():
            return s
        return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')

from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer

# regex patterns for cleaning
NEWLINE_RE = re.compile(r"(\n)")
HASHTAG_AT_RE = re.compile(r"([#@\u0040\uFF20\uFE6B])")
HASHTAG_ATS_WORD_RE = re.compile(r"([#@]\w+)")
EMOJIS_RE = re.compile(r"([\u2600-\u27FF])")
URL_RE = re.compile(r"(?:\w+:\/{2})?(?:www)?(?:\.)?([a-z\d]+)(?:\.)([a-z\d\.]{2,})(\/[a-zA-Z\/\d]+)?")

# punctuation pattern (keeps apostrophes so tokenizers can handle contractions)
PUNCTUATION_RE = re.compile(r"[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+")
STRAY_APOSTROPHE_RE = re.compile(r"(?<!\w)'|'(?!\w)")
WHITESPACE_RE = re.compile(r"(\s{2,})")
STRUCTURE_TAGS_RE = re.compile(r"\[.*?\]")

CONTRACTION_MAP = {
    "'m": "am", "n't": "not", "'s": "is", "'ve": "have",
    "'ll": "will", "'d": "would", "'re": "are", "can't": "cannot", "won't": "will not"
}

_LEMMATIZER = None
_STEMMER = None
_DETOKENIZER = None
_TREEBANK_TOKENIZER = None
_STOPWORDS_SET = None

def get_stopwords():
    global _STOPWORDS_SET
    if _STOPWORDS_SET is None:
        try:
            _STOPWORDS_SET = set(nltk.corpus.stopwords.words("english"))
        except Exception:
            nltk.download('stopwords', quiet=True)
            try:
                _STOPWORDS_SET = set(nltk.corpus.stopwords.words("english"))
            except Exception:
                _STOPWORDS_SET = set()
    return _STOPWORDS_SET

def get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        try:
            _LEMMATIZER = WordNetLemmatizer()
            _LEMMATIZER.lemmatize("test")
        except Exception:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER

@functools.lru_cache(maxsize=100000)
def _cached_lemmatize(token, pos):
    return get_lemmatizer().lemmatize(token, pos)

def get_stemmer():
    global _STEMMER
    if _STEMMER is None:
        _STEMMER = PorterStemmer()
    return _STEMMER

def get_detokenizer():
    global _DETOKENIZER
    if _DETOKENIZER is None:
        _DETOKENIZER = TreebankWordDetokenizer()
    return _DETOKENIZER

def get_tokenizer():
    global _TREEBANK_TOKENIZER
    if _TREEBANK_TOKENIZER is None:
        _TREEBANK_TOKENIZER = TreebankWordTokenizer()
    return _TREEBANK_TOKENIZER



def regex_cleaner(raw_text, 
                  no_emojis=True, 
                  no_hashtags=True,
                  hashtag_retain_words=True, 
                  no_newlines=True,
                  no_urls=True,
                  no_punctuation=True,
                  strip_structure_tags=True):
    """
    Cleans raw text using regex (emojis, URLs, hashtags, punctuation, bracketed tags).
    """
    if not isinstance(raw_text, str) or not raw_text:
        return ""

    clean_text = raw_text

    # remove structure tags like [Chorus], [Verse]
    if strip_structure_tags:
        clean_text = STRUCTURE_TAGS_RE.sub(" ", clean_text)

    if no_emojis:
        clean_text = EMOJIS_RE.sub("", clean_text)

    if no_hashtags:
        if hashtag_retain_words:
            clean_text = HASHTAG_AT_RE.sub("", clean_text)
        else:
            clean_text = HASHTAG_ATS_WORD_RE.sub("", clean_text)
        
    if no_newlines:
        clean_text = NEWLINE_RE.sub(" ", clean_text)

    if no_urls:
        clean_text = URL_RE.sub("", clean_text)
    
    if no_punctuation:
        clean_text = PUNCTUATION_RE.sub("", clean_text)
        clean_text = STRAY_APOSTROPHE_RE.sub("", clean_text)

    clean_text = WHITESPACE_RE.sub(" ", clean_text).strip()
    return clean_text


def extract_all_choruses(lyrics, fallback_lines=6):
    """
    Extracts all chorus sections from lyrics.
    Falls back to the first few lines if no [Chorus] tag is present.
    """
    if not isinstance(lyrics, str) or not lyrics.strip():
        return ""

    matches = re.findall(r'\[Chorus.*?\](.*?)(?=\[|$)', lyrics, flags=re.IGNORECASE | re.DOTALL)
    chorus_text = " ".join(m.strip() for m in matches if m.strip())
    
    if len(chorus_text) > 10:
        return chorus_text

    # fallback to first lines if no chorus tag found
    first_lines = lyrics.strip().split('\n')[:fallback_lines]
    return ' '.join(first_lines)


def lemmatize_all(token, list_pos=["n", "v", "a", "r", "s"]):
    """
    Lemmatizes token across specified POS tags.
    """
    for pos in list_pos:
        token = _cached_lemmatize(token, pos)
    return token


def main_pipeline(raw_text, 
                  print_output=False, 
                  no_stopwords=True,
                  custom_stopwords=None,
                  convert_diacritics=True, 
                  lowercase=True, 
                  lemmatized=True,
                  list_pos=["v"],
                  stemmed=False, 
                  pos_tags_list="no_pos",
                  tokenized_output=False,
                  max_words=None,
                  **kwargs):
    """
    Main text preprocessing pipeline.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return [] if tokenized_output else ""

    clean_text = regex_cleaner(raw_text, **kwargs)

    # tokenize
    try:
        tokenized_text = get_tokenizer().tokenize(clean_text)
    except Exception:
        tokenized_text = clean_text.split()

    # cap max word count if specified
    if max_words is not None and max_words > 0:
        tokenized_text = tokenized_text[:max_words]

    # expand common contractions
    tokenized_text = [CONTRACTION_MAP.get(token.lower(), token) for token in tokenized_text]

    # convert accented characters
    if convert_diacritics:
        tokenized_text = [unidecode(token) for token in tokenized_text]

    # stopwords removal
    if no_stopwords:
        stopwords_set = get_stopwords()
        tokenized_text = [item for item in tokenized_text if item.lower() not in stopwords_set]
        if custom_stopwords:
            c_set = set(custom_stopwords)
            tokenized_text = [item for item in tokenized_text if item.lower() not in c_set]

    # optional POS tagging
    if pos_tags_list in ("pos_list", "pos_tuples", "pos_dictionary"):
        try:
            pos_tuples = nltk.tag.pos_tag(tokenized_text)
            pos_tags = [pos[1] for pos in pos_tuples]
        except Exception:
            pos_tuples = [(t, "NN") for t in tokenized_text]
            pos_tags = ["NN" for _ in tokenized_text]

    # lowercase
    if lowercase:
        tokenized_text = [item.lower() for item in tokenized_text]

    # lemmatization
    if lemmatized:
        tokenized_text = [lemmatize_all(token, list_pos=list_pos) for token in tokenized_text]

    # stemming
    if stemmed:
        tokenized_text = [get_stemmer().stem(token) for token in tokenized_text]

    if print_output:
        print("Raw text:", raw_text[:100])
        print("Tokens:", tokenized_text[:20])

    if pos_tags_list == "pos_list":
        return (tokenized_text, pos_tags)
    elif pos_tags_list == "pos_tuples":
        return pos_tuples

    if tokenized_output:
        return tokenized_text
    else:
        return str(get_detokenizer().detokenize(tokenized_text))


def clean_for_classification(text, max_words=1000):
    """
    Preprocessing for classification: removes stopwords and punctuation,
    lowercases, converts diacritics, and lemmatizes verbs.
    """
    return main_pipeline(
        text,
        print_output=False,
        no_stopwords=True,
        convert_diacritics=True,
        lowercase=True,
        lemmatized=True,
        list_pos=["v"],
        stemmed=False,
        tokenized_output=False,
        max_words=max_words,
        no_punctuation=True,
        strip_structure_tags=True
    )


def clean_for_sentiment(text, max_words=1000):
    """
    Preprocessing for sentiment analysis: keeps stopwords, punctuation,
    and original casing.
    """
    return main_pipeline(
        text,
        print_output=False,
        no_stopwords=False,
        no_punctuation=False,
        convert_diacritics=True,
        lowercase=False,
        lemmatized=False,
        stemmed=False,
        tokenized_output=False,
        max_words=max_words,
        strip_structure_tags=True
    )


def cooccurrence_matrix_sentence_generator(preproc_sentences, sentence_cooc=False, window_size=5):
    """
    Builds a co-occurrence matrix using a sliding window or sentence level.
    """
    co_occurrences = defaultdict(Counter)

    if sentence_cooc:
        for sentence in tqdm(preproc_sentences, desc="Sentence co-occurrence"):
            for token_1 in sentence:
                for token_2 in sentence:
                    if token_1 != token_2:
                        co_occurrences[token_1][token_2] += 1
    else:
        for sentence in tqdm(preproc_sentences, desc="Sliding window co-occurrence"):
            for i, word in enumerate(sentence):
                for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                    if i != j:
                        co_occurrences[word][sentence[j]] += 1

    unique_words = list(set([word for sentence in preproc_sentences for word in sentence]))
    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)
    word_index = {word: idx for idx, word in enumerate(unique_words)}

    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word]][word_index[neighbor]] = count

    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=1)
    co_matrix_df = co_matrix_df.reindex(co_matrix_df.sum().sort_values(ascending=False).index, axis=0)

    return co_matrix_df


def plot_term_frequency(df, nr_terms=20, df_name="", show=False):
    """
    Plots horizontal barplot for the top N most frequent terms.
    """
    plt.figure(figsize=(10, 6))
    top_df = df.head(nr_terms)
    term_col = top_df.columns[0]
    count_col = top_df.columns[1]
    
    ax = sns.barplot(data=top_df, x=count_col, y=term_col, palette="mako")
    plt.title(f"Top {nr_terms} Terms: {df_name}", fontsize=14, weight="bold")
    plt.xlabel("Frequency / Weight")
    plt.ylabel("Term")
    plt.tight_layout()
    if show:
        plt.show()
    return ax.figure
