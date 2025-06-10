import json
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

nltk.download('stopwords')

# 定义文件路径
raw_article_path = r"D:\project(py)\Attention_Word_embedding\过去的、不成熟的观点\demo全部\embeddings\ABR_raw.txt"
output_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\output\tfidf_scores.json"

# 读取原文章
with open(raw_article_path, 'r', encoding='utf-8') as file:
    raw_article = file.read()

# 数据预处理：删除符号、数字和停用词
def preprocess(text):
    text = re.sub(r'\d+', '', text)  # 删除数字
    text = re.sub(r'[^\w\s]', '', text)  # 删除符号
    text = text.lower()  # 转换为小写
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # 删除停用词
    return ' '.join(tokens)

preprocessed_article = preprocess(raw_article)

# 使用 TF-IDF 计算单词的重要性
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([preprocessed_article])
tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

# 输出调试信息
logger.debug(
    "Sample TF-IDF scores: %s", list(tfidf_scores.items())[:10]
)

# 保存结果到文件
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(tfidf_scores, output_file, ensure_ascii=False, indent=4)

logger.info("TF-IDF scores saved to %s", output_path)
