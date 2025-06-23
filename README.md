# Supply Chain Risk Measurement

This project builds supply chain and risk vocabularies using word embeddings and applies them to earnings call transcripts,which provides a simple script to score each transcript based on the proximity of supply chain terms to risk-related terms.

## Usage

```bash
pip install -r requirements.txt
pip install -e .
# optional: install nltk for helper scripts
pip install nltk
python -m nltk.downloader stopwords punkt
run-measure examples/EarningCall_demo.xlsx --expand
```
The last two commands are needed if you plan to run `run-measure` or other tools that rely on NLTK. They also download the required datasets (`stopwords` and `punkt`).

### NLTK data

If you installed `nltk` but the automatic download did not run, open a Python shell and execute:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Input file

`scripts/run_measure.py` expects an Excel file containing earnings call text. A
small example `examples/EarningCall_demo.xlsx` and a larger sample `demo_57条.xls` are
provided in the repository.

### Dataset

The example transcripts were retrieved from the S&P Capital IQ Transcripts database. Obtain the full dataset through Capital IQ and export the calls you need to Excel format before running the scripts.

The `--expand` flag uses a small pre-trained GloVe model to expand the initial word lists located in `expanded_sc_words.txt` and `expanded_risk_words.txt`. The resulting scores are saved to `scores.xlsx`. This repository also includes an example output `meeting_scores_57.xlsx` generated from the `demo_57条.xls` dataset.

### GloVe model

The file `glove/glove.6B.300d.txt` in this repository is only a placeholder.
Download `glove.6B.zip` from the [Stanford NLP site](https://nlp.stanford.edu/data/glove.6B.zip)
and extract it:

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

Move `glove.6B.300d.txt` from the extracted folder into the `glove/` directory,
replacing the placeholder. A tiny sample generated from the publicly available
`glove-wiki-gigaword-50` model is provided in `glove/sample_glove.txt` for quick
tests.
### BERT model

Some example scripts expect a Hugging Face model stored in `bert-base-uncased/`.
Use `transformers` to fetch the pretrained weights and tokenizer:

```python
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('bert-base-uncased', cache_dir='bert-base-uncased')
AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='bert-base-uncased')
```

This will place all required files in the `bert-base-uncased/` directory.

## Files

- `risk_measure` – core functions for vocabulary expansion and scoring (installable package).
- `scripts/run_measure.py` – thin wrapper calling `risk_measure.cli`.
- `expanded_sc_words.txt`, `expanded_risk_words.txt` – editable seed vocabularies.

## Command line usage

After installation the `run-measure` command becomes available:

```bash
run-measure examples/EarningCall_demo.xlsx --expand
```

Use `--sc-words`, `--risk-words` and `--output` to customize the input and output files.

## Testing

Install `pytest` along with the project requirements and run the suite from the
repository root:

```bash
pip install -r requirements.txt pytest
pytest
```

## Future Improvements

This project currently scores supply chain risk exposure by measuring the proximity between supply chain and risk-related terms. To enhance accuracy and contextual relevance, several methodological improvements are under consideration:

### 1. Hybrid Word Importance Scoring

Combine two complementary methods for word importance:

* **Transformer Attention**: captures contextual salience of words.
* **TextRank**: evaluates structural centrality in the document.

Formula:

$$
Importance(w) = α · Attention(w) + (1 - α) · TextRank(w)
$$



This balances local semantic relevance and global structural influence. The hyperparameter α ∈ \[0, 1] can be tuned.


### 2. Contextual Embedding Similarity

Use contextualized embeddings (e.g., BERT, RoBERTa) instead of static ones like GloVe:

$$
Similarity(w, c) = cos(E_BERT(w), E_BERT(c))
$$

This enables term importance to adapt dynamically based on context, capturing industry-specific or ambiguous usage.



### 3. Dependency-Aware Proximity

Replace linear distance with syntactic dependency tree distance to compute proximity-based scores:

$$
ProximityScore(w_i, w_j) = 1 / (1 + λ · DepTreeDist(w_i, w_j))
$$

This allows better recognition of meaningful pairings like “supplier delay” or “inventory shortage” beyond word order.


### 4. Sentence-Level Risk Aggregation

Aggregate word-level importance into sentence-level risk intensity:

$$
SentenceRisk(s) = (1 / |s|) · Σ Importance(w) · Presence(w)
$$

This enables sentence-based ranking or highlighting of risk-heavy sentences in long transcripts.



# 供应链风险测度

本项目使用词向量扩展供应链及风险词典，并对电话会议文本计算风险得分。
示例脚本如下：

```bash
pip install -r requirements.txt
pip install -e .
# 当使用例子脚本时才需要 nltk
pip install nltk
python -m nltk.downloader stopwords punkt
python scripts/run_measure.py examples/EarningCall_demo.xlsx --expand
```
加上 `--expand` 参数后，会利用预训练的 GloVe 模型扩充词典，并把结果保存到 `scores.xlsx`。仓库中附带 `demo_57条.xls` 输入文件及其生成的示例结果 `meeting_scores_57.xlsx` 供参考。

### 数据集

示例电话会议文本来自 S&P Capital IQ 的 Transcripts 数据库。请从 Capital IQ 导出所需会议记录并另存为 Excel 文件后再运行脚本。

仓库中的 `glove/glove.6B.300d.txt` 只是**占位符**。若要使用完整模型，可从 [Stanford NLP 网站](https://nlp.stanford.edu/data/glove.6B.zip) 下载并解压，
请**用真实的 `glove.6B.300d.txt` 覆盖该文件**。项目中附带 `glove/sample_glove.txt` 供快速测试使用。
### BERT模型

部分脚本需要在 `bert-base-uncased/` 目录下存放完整的预训练 BERT 模型，可使用 `transformers` 从 Hugging Face 下载后放入该目录。

## 未来可扩展方向

本项目当前通过计算供应链与风险词的距离来度量文本中的风险暴露程度。为了提升语义精度和上下文适应能力，后续可考虑如下改进：

### 1. 混合词语重要性评分

将两类互补的方法结合以提升词语显著性识别：

* **Transformer 注意力**：捕捉词语在语境中的权重；
* **TextRank 图算法**：识别文档中结构上“中心”的词语。

数学表达为：

$$
Importance(w) = α · Attention(w) + (1 - α) · TextRank(w)
$$

其中 $α∈\[0,1]$ 为调节参数，可通过实验确定最优值。



### 2. 上下文嵌入相似度

用上下文相关的 BERT 嵌入替换静态词向量（如 GloVe）以计算词语与核心概念词的语义相关度：

$$
Similarity(w, c) = cos(E_BERT(w), E_BERT(c))
$$

相比静态词向量，这种方式能捕捉语境变化，适用于行业术语或语义多义的场景。



### 3. 基于依存结构的词距权重

用句法依存树上的路径长度替代线性距离：

$$
ProximityScore(w_i, w_j) = 1 / (1 + λ · DepTreeDist(w_i, w_j))
$$

能够更好识别语义上紧密相关的词对，如“供应商延误”、“库存短缺”等。


### 4. 句子级风险聚合

将词语重要性聚合为句子风险强度评分：

$$
SentenceRisk(s) = (1 / |s|) · Σ Importance(w) · Presence(w)
$$

适用于构建句子风险排序或高亮工具，提升会议文本风险定位的可读性。


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
