# Supply Chain Risk Measurement Based on Text Analysis

## Project Motivation

In the modern global economy, the stability of supply chains is critical to corporate performance. Traditional risk assessment methods often rely on lagging financial reports. This project aims to leverage Natural Language Processing (NLP) to develop a more timely and sensitive quantitative measure of supply chain risk by analyzing the language used by executives in corporate earnings calls, providing valuable decision support for investors and managers.

## Core Methodology

The core algorithm quantifies risk exposure by measuring the co-occurrence of "supply chain" and "risk" related terms within a specific proximity window in a given text. A higher frequency of these terms appearing close together indicates a stronger association between supply chain issues and risk, resulting in a higher score.

The risk score for each earnings call transcript (k) is calculated using the following formula:

$SCRisk_k = \frac{1}{N_k} \sum_{\forall i \in \text{SupplyChainWords}} \sum_{\forall j \in \text{RiskWords}} F_{i,k} \times \mathbf{1}[|p_i - p_j| \le 10]$


Where:
* $SCRisk_k$: The final calculated supply chain risk score for transcript $k$.
* $N_k$: A normalization factor, such as the total number of words in transcript $k$, to ensure scores are comparable across documents of different lengths.
* $F_{i,k}$: The frequency of the supply chain-related term $i$ in transcript $k$.
* $p_i$ and $p_j$: The position (index) of words $i$ and $j$ in the transcript.
* $\mathbf{1}[|p_i - p_j| \le 10]$: An indicator function that equals 1 if a risk word $j$ appears within a 10-word window (before or after) a supply chain word $i$, and 0 otherwise.

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

The example transcripts were retrieved from the *S&P Capital IQ Transcripts* database. Obtain the full dataset through Capital IQ and export the calls you need to Excel format before running the scripts.

The `--expand` flag uses a small pre-trained *GloVe* model to expand the initial word lists located in `expanded_sc_words.txt` and `expanded_risk_words.txt`. The resulting scores are saved to `scores.xlsx`. This repository also includes an example output `meeting_scores_57.xlsx` generated from the `demo_57条.xls` dataset.

### *GloVe* model

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
### *BERT* model

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

### Future Improvements

The current proximity-based scoring method is effective but can be enhanced to better capture contextual relevance. Future work will focus on developing a more sophisticated hybrid word importance scoring model.

**Hybrid Word Importance Scoring**

This approach combines two complementary methods to create a more robust word salience score:

* **Transformer Attention**: Excels at capturing local, context-dependent word importance.
* **TextRank**: Identifies words that are globally, structurally central to the entire document.

The proposed formula is:
$$\text{Importance}(w) = \alpha \cdot \text{Attention}(w) + (1 - \alpha) \cdot \text{TextRank}(w)$$

By combining these, we aim to create a score that is sensitive to both local semantic context and global document structure. The hyperparameter $\alpha \in [0, 1]$ can be tuned to balance the two components.



# 供应链风险测度

## 项目动机

在现代全球经济中，供应链的稳定性对企业至关重要。传统的风险评估往往依赖于滞后的财务报告。本项目旨在利用自然语言处理技术，通过实时分析企业高管在财报电话会议中的语言，来构建一个更即时、更敏感的供应链风险量化指标，为投资者和管理者提供决策支持。

## 核心方法论

本项目的核心算法通过计算“供应链”与“风险”相关词汇在给定文本中特定窗口内的共现频率来量化风险暴露。相关词汇在文本中出现的位置越近、频率越高，则意味着供应链问题与风险的关联性越强，风险得分也越高。

每个财报电话会议文本（k）的风险得分由以下公式计算：

$SCRisk_k = \frac{1}{N_k} \sum_{\forall i \in \text{供应链词汇}} \sum_{\forall j \in \text{风险词汇}} F_{i,k} \times \mathbf{1}[|p_i - p_j| \le 10]$

其中:
* $SCRisk_k$: 文本 $k$ 的最终供应链风险得分。
* $N_k$: 归一化因子，例如文本 $k$ 的总词数，以确保不同长度文档的得分具有可比性。
* $F_{i,k}$: 供应链相关词汇 $i$ 在文本 $k$ 中出现的频率。
* $p_i$ 和 $p_j$: 词汇 $i$ 和 $j$ 在文本中的位置（索引）。
* $\mathbf{1}[|p_i - p_j| \le 10]$: 指示函数。当一个风险词汇 $j$ 出现在一个供应链词汇 $i$ 前后10个词的窗口内时，其值为1，否则为0。

## 使用说明

示例脚本如下：

```bash
pip install -r requirements.txt
pip install -e .
# 当使用例子脚本时才需要 nltk
pip install nltk
python -m nltk.downloader stopwords punkt
python scripts/run_measure.py examples/EarningCall_demo.xlsx --expand
```
加上 `--expand` 参数后，会利用预训练的 *GloVe* 模型扩充词典，并把结果保存到 `scores.xlsx`。仓库中附带 `demo_57条.xls` 输入文件及其生成的示例结果 `meeting_scores_57.xlsx` 供参考。

### 数据集

示例电话会议文本来自 *S&P Capital IQ* 的 Transcripts 数据库。请从 *Capital IQ* 导出所需会议记录并另存为 Excel 文件后再运行脚本。

仓库中的 `glove/glove.6B.300d.txt` 只是**占位符**。若要使用完整模型，可从 [Stanford NLP 网站](https://nlp.stanford.edu/data/glove.6B.zip) 下载并解压，
请**用真实的 `glove.6B.300d.txt` 覆盖该文件**。项目中附带 `glove/sample_glove.txt` 供快速测试使用。

### BERT模型

部分脚本需要在 `bert-base-uncased/` 目录下存放完整的预训练 BERT 模型，可使用 `transformers` 从 Hugging Face 下载后放入该目录。

## 未来可扩展方向

当前基于词语距离的评分方法是有效的，但仍有提升空间以更好地捕捉上下文关联。未来的工作将聚焦于构建一个更精密的混合词语重要性评分模型。

**混合词语重要性评分**

该方法旨在结合两种互补的技术，以创建更鲁棒的词语显著性评分：

* **Transformer 注意力**：擅长捕捉词语在**局部上下文**中的重要性。
* **TextRank 图算法**：能识别出在**全局文档结构**中处于中心的词语。

其数学表达为：
$$\text{Importance}(w) = \alpha \cdot \text{Attention}(w) + (1 - \alpha) \cdot \text{TextRank}(w)$$

通过融合两者，我们期望得到一个既能感知局部语义，又能把握全局重点的评分体系。其中超参数 $\alpha \in [0, 1]$ 可通过实验确定最优值。


