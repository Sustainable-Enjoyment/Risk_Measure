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


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
