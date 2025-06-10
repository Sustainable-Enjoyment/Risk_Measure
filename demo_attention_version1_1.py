# scrisk += (attention_i * attention_j)
# 10-->15
import json
import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# 路径
tokens_dir = r"D:\project(py)\Attention_Word_embedding\过去的、不成熟的观点\demo全部\tokens"
output_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\output"
model_path = r"D:\database\bert-base-uncased"
risk_dict_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\RISK\expanded_risk_words.txt"
supply_chain_dict_path = r"D:\project(py)\Attention_Word_embedding\现在的、正在尝试的做法\SC\expanded_sc_words.txt"

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path, output_attentions=True)

# 读取词典
def load_dict(file_path):
    """读取词典"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

# 加载风险和供应链词典
risk_dict = load_dict(risk_dict_path)
supply_chain_dict = load_dict(supply_chain_dict_path)

def load_tokens(file_path):
    """读取分词（json）"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_attention_scores(tokens, model, tokenizer, max_length=512, stride=256):
    """每个token的注意力得分"""
    all_attention_scores = np.zeros(len(tokens))
    all_attention_counts = np.zeros(len(tokens))

    # 窗口滑动
    for i in range(0, len(tokens), stride):
        window = tokens[i:i + max_length]
        if len(window) < 10:
            continue  # 跳过非常短的窗口
        inputs = tokenizer(window, return_tensors='pt', is_split_into_words=True, padding='max_length', truncation=True, max_length=max_length)
        outputs = model(**inputs)

        # 汇总所有层和头的注意力得分
        layer_attentions = torch.stack(outputs.attentions).mean(dim=0)  # 形状: (batch_size, num_heads, seq_length, seq_length)
        head_mean_attentions = layer_attentions.mean(dim=1)  # 形状: (batch_size, seq_length, seq_length)
        token_attention_scores = head_mean_attentions.mean(dim=-1).squeeze(0).detach().numpy()[:len(window)]  # 形状: (seq_length,)

        for idx, score in enumerate(token_attention_scores):
            actual_idx = i + idx
            if actual_idx < len(tokens):
                all_attention_scores[actual_idx] += score
                all_attention_counts[actual_idx] += 1

    # 平均注意力得分
    averaged_attention_scores = all_attention_scores / np.maximum(all_attention_counts, 1)
    return averaged_attention_scores

def calculate_scrisk(tokens, attention_scores, risk_dict, supply_chain_dict):
    """计算测度"""
    scrisk = 0
    risk_tokens = [(i, token) for i, token in enumerate(tokens) if token in risk_dict]
    supply_chain_tokens = [(i, token) for i, token in enumerate(tokens) if token in supply_chain_dict]

    for i, token_i in risk_tokens:
        for j, token_j in supply_chain_tokens:
            if abs(i - j) <= 15:
                attention_i = attention_scores[i]
                attention_j = attention_scores[j]

                scrisk += (attention_i * attention_j)  # 计算SCRisk
    return scrisk

def process_file(file_path, model, tokenizer):
    """计算文件的SCRisk"""
    tokens = load_tokens(file_path)
    attention_scores = get_attention_scores(tokens, model, tokenizer)
    scrisk = calculate_scrisk(tokens, attention_scores, risk_dict, supply_chain_dict)
    return scrisk

def main():
    # 输出路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 遍历
    results = {}
    for file_name in os.listdir(tokens_dir):
        # 跳过非json文件
        if file_name.endswith('_tokens.json'):
            # 计算SCRisk
            file_path = os.path.join(tokens_dir, file_name)
            scrisk = process_file(file_path, model, tokenizer)
            key = file_name.replace('_tokens.json', '')
            results[key] = scrisk * 1e6  # 放大10的6次方

    # 五位有效数字
    results = {k: round(v, 5) for k, v in results.items()}

    output_file = os.path.join(output_path, "demo_attention_1.1_results.json")
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, indent=4)
    print(f"结果已保存到 {output_file}")

main()
