import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取
file_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\demo_57条_dic.xlsx"
df = pd.read_excel(file_path)

# 会议内容
contents = df['Formatted_content'].tolist()

# GloVe
glove_path = r"D:\database\GloVe\6B\glove.6B.300d.txt"
def load_glove_embeddings(glove_path):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

embeddings_index = load_glove_embeddings(glove_path)

# 预处理：分词、移除停用词
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 分词
    words = word_tokenize(text)
    # 移除停用词和非字母词
    words = [word for word in words if word.isalpha() and word.lower() not in stop_words]
    return words

preprocessed_contents = [' '.join(preprocess_text(content)) for content in contents]

# 计算TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_contents)
feature_names = vectorizer.get_feature_names_out()

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
document_embeddings = []
for i in range(len(preprocessed_contents)):
    doc_embedding = get_document_embedding(preprocessed_contents[i], embeddings_index, feature_names, tfidf_matrix[i].toarray()[0])
    document_embeddings.append(doc_embedding)

# 初始风险词库
initial_risk_words = ["risk", "crisis", "uncertainty", "loss", "hazard", "threat", "danger", "exposure", "vulnerability"]

# 获取初始风险词的词向量
initial_risk_word_embeddings = {}
for word in initial_risk_words:
    if word in embeddings_index:
        initial_risk_word_embeddings[word] = embeddings_index[word]

# 自动扩展风险词汇
expanded_risk_words = set(initial_risk_words)

def find_similar_words(word, embeddings_index, top_n=10):
    similar_words = []
    word_vector = embeddings_index[word]
    for candidate_word, candidate_vector in embeddings_index.items():
        similarity = cosine_similarity([word_vector], [candidate_vector])[0][0]
        similar_words.append((candidate_word, similarity))
    similar_words = sorted(similar_words, key=lambda x: x[1], reverse=True)[:top_n]
    return [w for w, s in similar_words]

# 为每个初始风险词找到相似词、扩展词库
for word in initial_risk_words:
    if word in embeddings_index:
        similar_words = find_similar_words(word, embeddings_index)
        expanded_risk_words.update(similar_words)

with open(r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\RISK\expanded_risk_words.txt", 'w') as f:
    for word in expanded_risk_words:
        f.write(f"{word}\n")

print("Expanded risk word list has been saved to expanded_risk_words.txt")
