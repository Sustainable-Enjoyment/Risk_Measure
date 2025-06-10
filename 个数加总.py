import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 加载词典
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return set(words)

risk_dict_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\RISK\expanded_risk_words.txt"
sc_dict_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\SC\expanded_sc_words.txt"
risk_words = load_word_list(risk_dict_path)
sc_words = load_word_list(sc_dict_path)

# 读取
file_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\demo_57条_dic.xlsx"
df = pd.read_excel(file_path)
# 提取会议内容
contents = df['Formatted_content'].tolist()


# 预处理
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # 分词
    words = word_tokenize(text)
    # 转换为小写、移除停用词等
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return words


preprocessed_contents = [preprocess_text(content) for content in contents]


def calculate_attention_scores(words, risk_words, sc_words, window_size=10):
    total_score = 0

    for i, word in enumerate(words):
        if word in risk_words:
            risk_scores = []
            sc_scores = []
            window_start = max(0, i - window_size)
            window_end = min(len(words), i + window_size + 1)
            window_words = words[window_start:window_end]

            for w in window_words:
                if w in risk_words:
                    risk_scores.append(w)
                if w in sc_words:
                    sc_scores.append(w)

            for risk_word in risk_scores:
                for sc_word in sc_scores:
                    total_score += 1  # 计算risk_word和sc_word的乘积

    if len(words) == 0:
        return 0
    return total_score / len(words)


scores = []
for content in preprocessed_contents:
    score = calculate_attention_scores(content, risk_words, sc_words)
    scores.append(score)

# 得分放在DataFrame
df['SCRisk_Score'] = scores

# 删除 列：
columns_to_keep = df.columns.difference(['Content', 'Formatted_content'])
df = df[columns_to_keep]

output_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\meeting_scores_57.xlsx"
df.to_excel(output_path, index=False)

print(f"Scores have been saved to {output_path}")
