# AGENTS.md

本仓库的主要代码位于 `src/risk_measure`，提供 `run-measure` 脚本用于计算供应链风险指标。  
在向本仓库提交变更时，请遵循以下规则。

## 开发和提交
- 所有代码需遵循 **PEP 8** 风格，建议使用 `black` 进行格式化。
- 重要的函数或脚本都应包含清晰的 docstring。
- 新增或修改功能时，请在 `tests/` 目录补充相应的 **pytest** 单元测试。
- 依赖库请在 `pyproject.toml` 及 `requirements.txt` 中同步更新。
- 提交信息请使用 **英文**，简洁描述本次变更内容。

## 测试
1. 安装依赖  
   ```bash
   pip install -r requirements.txt
