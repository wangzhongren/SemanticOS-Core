# SemanticOS-Core

语义操作系统（Semantic OS）的核心实现模块，基于 Qwen-0.6B 语言模型构建压缩语义内核，支持 4-bit 量化存储与语义寻址召回。

## 功能特性

- **语义压缩存储**：将文本规则通过语言模型编码为向量，并以 4-bit 量化形式压缩存储。
- **语义寻址召回**：对查询文本进行向量映射，通过余弦相似度在压缩内存中检索最匹配规则。
- **内核异常协议**：当匹配得分低于阈值（默认 0.70）时，触发《语义内核崩溃报告》，防止幻觉输出。
- **确定性分类**：严格遵循“分类即智能”原则，拒绝概率性推测。
- **灵活模型加载**：模型路径在运行时传入，不再硬编码于核心代码中。

## 模块结构

- `core/`：核心模块目录
  - `core.py`：核心类 `CompressedSemanticKernel` 实现，包含压缩、解压、存储与召回逻辑。
  - `__init__.py`：包初始化文件，支持直接从 `core` 导入。
- `test_core.py`：示例测试脚本，演示规则存入、精准召回与异常触发流程。

## 依赖

- Python ≥ 3.8
- PyTorch
- Transformers (Hugging Face)
- Qwen3-0.6B 模型（需放置于本地路径，如 `../diffusion2/Qwen/Qwen3-0.6B`）

## 使用示例

```python
from core import CompressedSemanticKernel

# 初始化时传入模型路径
MODEL_PATH = "path/to/your/model"
kernel = CompressedSemanticKernel(MODEL_PATH)

kernel.store_rule("Security: All encryption keys must be stored in HSM.", 0)
result, score = kernel.recall("Where should encryption keys be stored?")
```

## 设计依据

本实现参考语义操作系统理论框架，包括：
- 语义内存架构（Cognitive State Zone / Factual Payload Zone）
- Q-K-V 语义指令集（S-ISA）
- 确定性分类与内核异常处理机制

> 注：本项目为研究原型，模型路径和量化参数可根据实际环境调整。