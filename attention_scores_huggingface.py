import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

# 确保下载了所需的nltk数据包
nltk.download('punkt')
nltk.download('stopwords')

# 读取Excel文件
file_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\EarningCall_demo.xlsx"
df = pd.read_excel(file_path)

# 提取第二行的Formatted_content文本
text = df['Formatted_content'].iloc[1]

# 打印原始文本
# print("原始文本:")
# print(text)

# 预处理步骤
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# 预处理文本
preprocessed_words = preprocess_text(text)

# 打印预处理后的词汇
# print("预处理后的词汇:")
# print(preprocessed_words)
# print("剩余词汇数量:", len(preprocessed_words))

# 从本地目录加载分词器和模型
local_path = "D:/database/bert-base-uncased/"  # 使用正斜杠或双反斜杠
tokenizer = BertTokenizerFast.from_pretrained(local_path)
model = BertModel.from_pretrained(local_path, output_attentions=True)

# 对预处理后的文本进行BERT分词
inputs = tokenizer(preprocessed_words, return_tensors='pt', is_split_into_words=True, padding=True, truncation=True)

# 打印分词结果
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
# print("BERT分词结果:")
# print(tokens)
# print("分词数量:", len(tokens))

# 获取注意力向量
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions

# 验证注意力向量的形状
logger.debug("注意力向量形状: %s", attentions[0].shape)

# 计算每个词汇的平均注意力得分
attention_scores = attentions[-1].mean(dim=1).squeeze().detach().numpy()

# 检查注意力得分
logger.debug("注意力得分数量: %s", len(attention_scores))

# 合并子词的注意力得分
def merge_subword_attentions(tokens, attention_scores):
    merged_attentions = {}
    current_word = ""
    current_attention = 0
    token_count = 0

    for token, score in zip(tokens, attention_scores):
        if token.startswith("##"):
            current_word += token[2:]
            current_attention += score
            token_count += 1
        else:
            if current_word:
                merged_attentions[current_word] = current_attention / token_count
            current_word = token
            current_attention = score
            token_count = 1

    if current_word:
        merged_attentions[current_word] = current_attention / token_count

    return merged_attentions

# 合并注意力得分
merged_attentions = merge_subword_attentions(tokens, attention_scores)

# 打印合并后的词汇和对应的注意力得分
# print("合并后的词汇和对应的注意力得分:")
# for word, score in merged_attentions.items():
#     print(f"{word}: {score}")

# 保存结果到Excel文件
output_file = r"D:\project(py)\Attention_Word_embedding\Attention_Scores.xlsx"
df_output = pd.DataFrame(list(merged_attentions.items()), columns=['Word', 'Attention Score'])
df_output.to_excel(output_file, index=False)

logger.info("注意力得分已保存到 %s", output_file)
