"""Example script for computing word importance using a GloVe model."""

import json
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

nltk.download("stopwords")

# 定义文件路径
raw_article_path = (
    r"D:\project(py)\Attention_Word_embedding\过去的、不成熟的观点"
    r"\demo全部\embeddings\ABR_raw.txt"
)
output_path = (
    r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法"
    r"\output\word2vec_scores.json"
)
glove_model_path = r"path\to\glove\model"  # 修改为GloVe模型的路径

# 读取原文章
with open(raw_article_path, "r", encoding="utf-8") as file:
    raw_article = file.read()


# 数据预处理：删除符号、数字和停用词
def preprocess(text):
    text = re.sub(r"\d+", "", text)  # 删除数字
    text = re.sub(r"[^\w\s]", "", text)  # 删除符号
    text = text.lower()  # 转换为小写
    tokens = text.split()
    tokens = [
        word for word in tokens if word not in stopwords.words("english")
    ]  # 删除停用词
    return tokens


preprocessed_article = preprocess(raw_article)

# 使用 GloVe 计算单词的重要性
model = KeyedVectors.load_word2vec_format(glove_model_path, binary=False)

word_scores = {}
for word in preprocessed_article:
    if word in model:
        word_scores[word] = model.similarity(
            word,
            model.most_similar(positive=[word], topn=1)[0][0],
        )

# 输出调试信息
logger.debug("Sample Word2Vec scores: %s", list(word_scores.items())[:10])

# 保存结果到文件
with open(output_path, "w", encoding="utf-8") as output_file:
    json.dump(word_scores, output_file, ensure_ascii=False, indent=4)

logger.info("Word2Vec scores saved to %s", output_path)
