# attention_search&key&value
# 计算attention时出现溢出问题，解决方法：减去最大值
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


# 加载词典
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return set(words)


def load_data(excel_path: Path) -> pd.DataFrame:
    return pd.read_excel(excel_path)

stop_words = set()


def preprocess_text(text):
    # 分词
    words = word_tokenize(text)
    # 转换为小写，移除停用词和非字母词
    words = [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]
    return words


def preprocess_contents(contents):
    return [preprocess_text(content) for content in contents]


# 计算查询（Query）、键（Key）和值（Value）
def calculate_attention_matrices(words, d_k=64):
    np.random.seed(42)  # 设置随机种子以获得可重复的结果
    X = np.array([np.random.rand(d_k) for _ in words])
    W_Q = np.random.rand(d_k, d_k)
    W_K = np.random.rand(d_k, d_k)
    W_V = np.random.rand(d_k, d_k)
    Q = X.dot(W_Q)
    K = X.dot(W_K)
    V = X.dot(W_V)
    return Q, K, V


# 计算注意力得分
def attention(Q, K, V):
    d_k = Q.shape[1]
    scores = Q.dot(K.T) / np.sqrt(d_k)
    # 稳定Softmax计算，减去每行的最大值
    max_scores = np.max(scores, axis=1, keepdims=True)
    scores = scores - max_scores
    attention_weights = (
        np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    )
    attention_output = attention_weights.dot(V)
    return attention_weights, attention_output


# 计算每个会议的得分
def calculate_attention_scores(
    preprocessed_contents,
    risk_words,
    sc_words,
    d_k=64,
):
    total_scores = []
    for idx, words in enumerate(preprocessed_contents):
        if len(words) == 0:
            total_scores.append(0)
            continue

        Q, K, V = calculate_attention_matrices(words, d_k)
        attention_weights, _ = attention(Q, K, V)

        risk_indices = [
            i for i, word in enumerate(words) if word in risk_words
        ]
        sc_indices = [
            i for i, word in enumerate(words) if word in sc_words
        ]

        score = 0

        for ri in risk_indices:
            for si in sc_indices:
                if abs(ri - si) <= 10:
                    score += (
                        attention_weights[ri, si]
                        * attention_weights[si, ri]
                    )

        total_scores.append(score / len(words))

        # 调试信息
        logger.debug(
            "Document %s/%s", idx + 1, len(preprocessed_contents)
        )
        logger.debug("Risk indices: %s", risk_indices)
        logger.debug("SC indices: %s", sc_indices)
        logger.debug(
            "Attention weights sample: %s", attention_weights[:5, :5]
        )
        logger.debug("Score: %s", score)

    return total_scores


# 计算得分
def main():
    parser = argparse.ArgumentParser(
        description="Calculate supply chain risk scores using attention"
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("EarningCall_demo.xlsx"),
        help="Input Excel file",
    )
    parser.add_argument(
        "--risk-words",
        type=Path,
        default=Path("expanded_risk_words.txt"),
        help="Risk word list",
    )
    parser.add_argument(
        "--sc-words",
        type=Path,
        default=Path("expanded_sc_words.txt"),
        help="Supply chain word list",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("meeting_scores_attention.xlsx"),
        help="Output Excel file",
    )
    args = parser.parse_args()

    risk_words = load_word_list(args.risk_words)
    sc_words = load_word_list(args.sc_words)

    df = load_data(args.excel)
    contents = df["Formatted_content"].tolist()

    nltk.download("stopwords")
    nltk.download("punkt")
    global stop_words
    stop_words = set(stopwords.words("english"))

    preprocessed_contents = preprocess_contents(contents)

    scores = calculate_attention_scores(
        preprocessed_contents,
        risk_words,
        sc_words,
    )

    df["SCRisk_Score"] = scores
    columns_to_keep = df.columns.difference(["Content", "Formatted_content"])
    df = df[columns_to_keep]
    df.to_excel(args.output, index=False)
    logger.info("Scores have been saved to %s", args.output)


if __name__ == "__main__":
    main()
