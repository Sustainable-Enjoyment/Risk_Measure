import re
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
from gensim.models import KeyedVectors
from gensim.downloader import load as api_load


def load_glove(model_name: str = "glove-wiki-gigaword-50") -> KeyedVectors:
    """Return a pre-trained GloVe model.

    Parameters
    ----------
    model_name:
        Identifier of the model to load from ``gensim``'s downloader.

    Returns
    -------
    KeyedVectors
        The loaded word vector model ready for similarity queries.
    """
    return api_load(model_name)


def expand_vocabulary(initial: Iterable[str], embeddings: KeyedVectors, top_n: int = 10) -> Set[str]:
    """Expand a seed vocabulary using vector similarity.

    Parameters
    ----------
    initial:
        Words forming the starting vocabulary.
    embeddings:
        Pre-trained word vectors used to find similar terms.
    top_n:
        Number of neighbors to include for each seed word.

    Returns
    -------
    Set[str]
        The union of the ``initial`` words and their nearest neighbors.
    """
    expanded = set(initial)
    for word in initial:
        if word in embeddings:
            for similar, _ in embeddings.most_similar(word, topn=top_n):
                expanded.add(similar)
    return expanded


def load_word_list(path: Path) -> Set[str]:
    """Read a newline separated word list from disk.

    Parameters
    ----------
    path:
        Location of a UTF-8 encoded text file containing one word per line.

    Returns
    -------
    Set[str]
        A set of all non-empty lines stripped of surrounding whitespace.
    """
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def save_word_list(words: Iterable[str], path: Path) -> None:
    """Write a list of words to a newline separated text file.

    Parameters
    ----------
    words:
        Iterable of words to store. Duplicates are removed.
    path:
        Destination file to write in UTF-8 encoding.

    Returns
    -------
    None
        The function performs I/O for its side effect only.
    """
    path.write_text("\n".join(sorted(set(words))), encoding="utf-8")


def tokenize(text: str) -> List[str]:
    """Split raw text into lowercase tokens consisting only of letters.

    Parameters
    ----------
    text:
        Arbitrary input text which may contain punctuation or numbers.

    Returns
    -------
    List[str]
        Individual tokens extracted from ``text`` after basic cleaning.
    """
    text = re.sub(r"[^A-Za-z]+", " ", text).lower()
    return text.split()


def compute_score(tokens: List[str], sc_words: Set[str], risk_words: Set[str], window: int = 10) -> float:
    """Calculate the proportion of supply-chain terms near risk terms.

    Parameters
    ----------
    tokens:
        Tokenized transcript text.
    sc_words:
        Vocabulary of supply chain related words.
    risk_words:
        Vocabulary of risk related words.
    window:
        Number of words on each side of a supply-chain term to consider.

    Returns
    -------
    float
        Ratio of supply-chain tokens that appear within ``window`` words of at
        least one risk token.
    """
    count = 0
    for i, word in enumerate(tokens):
        if word in sc_words:
            start = max(0, i - window)
            end = i + window + 1
            window_tokens = tokens[start:end]
            if any(w in risk_words for w in window_tokens):
                count += 1
    return count / max(len(tokens), 1)


def score_transcripts(excel_path: Path, sc_words: Set[str], risk_words: Set[str], column: str = "Formatted_content") -> pd.DataFrame:
    """Score every transcript in an Excel file.

    Parameters
    ----------
    excel_path:
        Path to the Excel workbook containing a column with transcript text.
    sc_words:
        Vocabulary of supply chain terms.
    risk_words:
        Vocabulary of risk terms.
    column:
        Name of the column in ``excel_path`` that holds the raw text.

    Returns
    -------
    pandas.DataFrame
        The original dataframe with an added ``SCRisk_Score`` column.
    """
    df = pd.read_excel(excel_path)
    scores = []
    for text in df[column].fillna(""):
        tokens = tokenize(text)
        scores.append(compute_score(tokens, sc_words, risk_words))
    df["SCRisk_Score"] = scores
    return df


__all__ = [
    "load_glove",
    "expand_vocabulary",
    "load_word_list",
    "save_word_list",
    "tokenize",
    "compute_score",
    "score_transcripts",
]
