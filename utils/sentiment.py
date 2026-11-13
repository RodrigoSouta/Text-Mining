import re
import numpy as np
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer

from utils.pipeline import regex_cleaner

# section header such as [Chorus], [Verse 2: Artist], [Hook]
SECTION_RE = re.compile(r"^\s*\[(.*?)\]\s*$")
WORD_RE = re.compile(r"[a-z']+")

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05

_SIA = None


def get_sia():
    global _SIA
    if _SIA is None:
        _SIA = SentimentIntensityAnalyzer()
    return _SIA


def _section_type(header):
    header = header.lower()
    if 'chorus' in header or 'hook' in header or 'refrain' in header:
        return 'chorus'
    if 'verse' in header:
        return 'verse'
    return 'other'


def split_lines(lyrics, min_words=2):
    """
    Splits raw lyrics into cleaned lines, keeping the section type of each line.
    Punctuation, casing and stopwords are kept because VADER uses them.
    """
    if not isinstance(lyrics, str):
        return []
    lines, section = [], 'other'
    for raw in lyrics.split('\n'):
        header = SECTION_RE.match(raw)
        if header:
            section = _section_type(header.group(1))
            continue
        line = regex_cleaner(raw, no_punctuation=False, no_newlines=False)
        if len(line.split()) >= min_words:
            lines.append((line, section))
    return lines


def score_song(lyrics):
    """
    Scores every line of a song with VADER and aggregates to song level.
    Returns line-level summary statistics plus the song-level compound
    (whole text scored at once) for comparison.
    """
    sia = get_sia()
    lines = split_lines(lyrics)
    out = {'n_lines': len(lines)}
    if not lines:
        return out

    comp = np.array([sia.polarity_scores(l)['compound'] for l, _ in lines])
    sections = np.array([s for _, s in lines])
    third = max(1, len(comp) // 3)

    out.update({
        'mean_compound': comp.mean(),
        'pos_share': (comp >= POS_THRESHOLD).mean(),
        'neg_share': (comp <= NEG_THRESHOLD).mean(),
        'neu_share': ((comp > NEG_THRESHOLD) & (comp < POS_THRESHOLD)).mean(),
        'chorus_compound': comp[sections == 'chorus'].mean() if (sections == 'chorus').any() else np.nan,
        'verse_compound': comp[sections == 'verse'].mean() if (sections == 'verse').any() else np.nan,
        'first_third': comp[:third].mean(),
        'last_third': comp[-third:].mean(),
        'song_compound': sia.polarity_scores(' '.join(l for l, _ in lines))['compound'],
    })
    return out


def lexicon_word_counts(lyrics):
    """
    Counts the VADER lexicon words used in a song (lowercased).
    Used to see which words drive positive / negative scores in each genre.
    """
    lexicon = get_sia().lexicon
    if not isinstance(lyrics, str):
        return Counter()
    tokens = WORD_RE.findall(re.sub(r"\[.*?\]", " ", lyrics.lower()))
    return Counter(t for t in tokens if t in lexicon)
