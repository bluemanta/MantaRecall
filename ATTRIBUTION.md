# 第三方组件署名（ATTRIBUTION）

本项目（MantaRecall，Apache-2.0）使用了以下第三方组件。署名信息供学术榜复核与合规审查。

## 模型

- **cross-encoder/ms-marco-MiniLM-L-6-v2**
  - 用途：Search 精排（cross-encoder reranker，对 RRF 后 top 50 重排）
  - 来源：sentence-transformers / HuggingFace `cross-encoder/ms-marco-MiniLM-L6-v2`
   （经 ModelScope 镜像下载后 bake 进 Docker 镜像，运行时不依赖外网）
  - 许可：Apache-2.0
  - 说明：该模型在 MS MARCO passage ranking 数据集上训练；
    MS MARCO 数据集由 Microsoft 发布，仅供研究使用。

## 评测数据（仅离线评估用，不进入参赛服务）

- **LoCoMo**（Long Conversation Memory benchmark）
  - 用途：`eval/` 回放 harness 的离线评测数据（10 段长对话 / 1527 题）
  - 许可：CC BY-NC 4.0（仅非商业研究使用；仓库内附带数据文件）
  - 论文：Maharana et al., 2024

## 基础设施与依赖

- **PostgreSQL**（PostgreSQL License）：主数据库；`tsvector` 全文检索为原生功能
- **pgvector**（PostgreSQL License）：向量检索扩展（`CREATE EXTENSION vector`）
- **PyTorch**（BSD）：cross-encoder CPU 推理
- **sentence-transformers**（Apache-2.0）：cross-encoder 加载与推理
- **FastAPI / uvicorn / asyncpg / httpx / pydantic**（MIT / BSD）：Web 服务与数据访问
- **text-embedding-v4**：学术榜指定 embedding（1024 维），经兼容网关调用，
  属外部服务而非本仓库代码

## 备注

- 本仓库不包含任何官方 AML held-out 测试数据；公开 LoCoMo 仅用于离线方法验证。
- 依赖版本锁定见 `requirements.txt`；模型版本与哈希见部署记录。
