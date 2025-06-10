import json
import os
import re
import torch
import nltk
import numpy as np
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel

nltk.download('stopwords')

# 定义文件路径
tokenized_words_path = r"D:\project(py)\Attention_Word_embedding\过去的、不成熟的观点\demo全部\tokens\ABR_tokens.json"
raw_article_path = r"D:\project(py)\Attention_Word_embedding\过去的、不成熟的观点\demo全部\embeddings\ABR_raw.txt"
output_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\output\attention_测试.json"
word_order_path = r"/output/attention_测试_order.json"
index_scores_path = r"/output/attention_测试_scores.json"
model_path = r"D:\database\bert-base-uncased"

# 读取分词后的单词
with open(tokenized_words_path, 'r', encoding='utf-8') as file:
    tokenized_words = json.load(file)

# 输出调试信息
print(f"Tokenized words (sample): {tokenized_words[:10]}")

# 确保 tokenized_words 是单词列表而不是数值
if isinstance(tokenized_words[0], list):
    tokenized_words = [item for sublist in tokenized_words for item in sublist]

# 读取原文章
with open(raw_article_path, 'r', encoding='utf-8') as file:
    raw_article = file.read()0


# 数据预处理：删除符号、数字和停用词
def preprocess(text):
    text = re.sub(r'\d+', '', text)  # 删除数字
    text = re.sub(r'[^\w\s]', '', text)  # 删除符号
    text = text.lower()  # 转换为小写
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # 删除停用词
    return ' '.join(tokens)


preprocessed_article = preprocess(raw_article)

# 加载本地的BERT模型和Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path, output_attentions=True)

# 将文章分成多个小段，每段不超过512个词
max_length = 512
tokens = tokenizer.tokenize(preprocessed_article)
segments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

attention_scores = []
all_tokens = []
all_indices = []

# 对每个段落计算注意力得分
for segment_index, segment in enumerate(segments):
    inputs = tokenizer(" ".join(segment), return_tensors='pt', truncation=True, padding='max_length',
                       max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions[-1]
    attention = attention.mean(dim=1).squeeze().cpu().numpy()

    # 获取实际长度，排除填充的部分
    actual_length = (inputs['input_ids'] != tokenizer.pad_token_id).sum().item()

    # 仅保留实际单词部分的注意力得分
    trimmed_attention = attention[:actual_length, :actual_length]
    attention_scores.append(trimmed_attention)

    # 获取每个段落的分词并扩展到all_tokens和all_indices
    segment_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())[:actual_length]
    segment_indices = list(range(segment_index * max_length, segment_index * max_length + actual_length))
    all_tokens.extend(segment_tokens)
    all_indices.extend(segment_indices)

# 调试输出: 检查是否分段正确
print(f"Number of segments: {len(segments)}")
print(f"Number of tokens: {len(tokens)}")
print(f"Total length of all tokens: {len(all_tokens)}")
print(f"Sample all_tokens: {all_tokens[:50]}")

# 创建字典保存每个单词在不同位置的索引和注意力得分
word_attention_scores = {}
word_order = []
index_scores = {}

# 处理每个段落的注意力得分
for segment_idx, attention in enumerate(attention_scores):
    segment_tokens = all_tokens[segment_idx * max_length: (segment_idx + 1) * max_length]
    segment_indices = all_indices[segment_idx * max_length: (segment_idx + 1) * max_length]

    for i, token in enumerate(segment_tokens):
        word_order.append({"word": token, "index": segment_indices[i]})
        attention_scores_at_position = attention[i, :]
        attention_score = attention_scores_at_position[i]
        if segment_indices[i] not in index_scores:
            index_scores[segment_indices[i]] = []
        index_scores[segment_indices[i]].append(round(float(attention_score), 8))

# 保存结果到文件
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(word_attention_scores, output_file, ensure_ascii=False, indent=4)

with open(word_order_path, 'w', encoding='utf-8') as output_file:
    json.dump(word_order, output_file, ensure_ascii=False, indent=4)

with open(index_scores_path, 'w', encoding='utf-8') as output_file:
    json.dump(index_scores, output_file, ensure_ascii=False, indent=4)

print(f"Attention scores saved to {output_path}")
print(f"Word order saved to {word_order_path}")
print(f"Index scores saved to {index_scores_path}")
