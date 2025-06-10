import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def load_glove_embeddings(glove_path: Path):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

stop_words = set()

def preprocess_text(text):
    # 分词
    words = word_tokenize(text)
    # 移除停用词和非字母词
    words = [word for word in words if word.isalpha() and word.lower() not in stop_words]
    return words

def preprocess_contents(contents):
    return [' '.join(preprocess_text(content)) for content in contents]

# 计算TF-IDF
def compute_tfidf_matrix(preprocessed_contents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_contents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

# 生成文本的加权词向量
def get_document_embedding(doc, embeddings_index, feature_names, tfidf_vector):
    words = doc.split()
    doc_embedding = np.zeros(300)  # GloVe的维度是300
    for word in words:
        if word in embeddings_index:
            try:
                word_index = feature_names.tolist().index(word)
                word_tfidf = tfidf_vector[word_index]
                doc_embedding += embeddings_index[word] * word_tfidf
            except ValueError:
                continue
    return doc_embedding

# 计算每个文档的词向量
def compute_document_embeddings(preprocessed_contents, embeddings_index, feature_names, tfidf_matrix):
    document_embeddings = []
    for i in range(len(preprocessed_contents)):
        doc_embedding = get_document_embedding(
            preprocessed_contents[i],
            embeddings_index,
            feature_names,
            tfidf_matrix[i].toarray()[0],
        )
        document_embeddings.append(doc_embedding)
    return document_embeddings


def find_similar_words(word, embeddings_index, top_n=10):
    similar_words = []
    word_vector = embeddings_index[word]
    for candidate_word, candidate_vector in embeddings_index.items():
        similarity = cosine_similarity([word_vector], [candidate_vector])[0][0]
        similar_words.append((candidate_word, similarity))
    similar_words = sorted(similar_words, key=lambda x: x[1], reverse=True)[:top_n]
    return [w for w, s in similar_words]

# 为每个初始供应链词找到相似词并扩展词库
def expand_words(initial_sc_words, embeddings_index, top_n=10):
    expanded_sc_words = set(initial_sc_words)
    for word in initial_sc_words:
        if word in embeddings_index:
            similar_words = find_similar_words(word, embeddings_index, top_n)
            expanded_sc_words.update(similar_words)
    return expanded_sc_words


def save_words(words, path: Path):
    with open(path, 'w') as f:
        for word in words:
            f.write(f"{word}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Expand supply chain vocabulary using TF-IDF and GloVe"
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("EarningCall_demo.xlsx"),
        help="Input Excel file",
    )
    parser.add_argument(
        "--glove",
        type=Path,
        default=Path("glove/glove.6B.300d.txt"),
        help="Path to GloVe embeddings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("expanded_sc_words.txt"),
        help="Output word list file",
    )
    args = parser.parse_args()

    df = load_excel(args.excel)
    contents = df["Formatted_content"].tolist()

    nltk.download("stopwords")
    nltk.download("punkt")
    global stop_words
    stop_words = set(stopwords.words("english"))

    preprocessed_contents = preprocess_contents(contents)
    tfidf_matrix, feature_names = compute_tfidf_matrix(preprocessed_contents)
    embeddings_index = load_glove_embeddings(args.glove)

    initial_sc_words = [
        "supply",
        "chain",
        "logistics",
        "procurement",
        "supplier",
        "inventory",
        "distribution",
        "manufacturing",
        "sourcing",
        "warehousing",
    ]

    _ = compute_document_embeddings(
        preprocessed_contents, embeddings_index, feature_names, tfidf_matrix
    )
    expanded_sc_words = expand_words(initial_sc_words, embeddings_index)
    save_words(expanded_sc_words, args.output)
    logger.info(
        "Expanded supply chain word list has been saved to %s", args.output
    )


if __name__ == "__main__":
    main()
