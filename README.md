# MantaRecall

Agent 长期记忆参赛系统（AML 第二轮 · 文本赛道 · 学术榜），v0.1.0。

FastAPI + pgvector 实现的 Add/Search 同步 API 骨架，开源方法榜参赛用。
**学术榜硬约束**：Embedding 必须用 `text-embedding-v4`（1024 维），LLM 相关组件必须用 `gpt-4o-mini`，Reranker 不限。

## 1. 接口契约（2026-10-01 核验）

| 接口 | 鉴权 | 语义 |
|---|---|---|
| `GET /health` | 无需 | 返回 2xx 即就绪 |
| `POST /add` | 需要 | 同步；HTTP 200 = 全部消息已持久化且立即可搜；响应原样回传 `request_id` / `user_id` / `session_id`；相同 `request_id`+相同内容→幂等 200；相同 `request_id`+不同内容→409 |
| `POST /search` | 需要 | 返回 `{"data": [...]}`，按相关性排序，数量不超过 `top_k`（正式评测 `top_k=100`）；**只返回记忆证据，不生成答案**；`user_id` 是唯一的检索隔离边界 |

鉴权三选一：`X-Api-Key: <key>` / `Authorization: Bearer <key>` / `Authorization: Token <key>`。

Add 请求体：`{"request_id","messages":[{"role","content","timestamp(毫秒)"}],"user_id","session_id"}`
Search 请求体：`{"query","options(可选)","user_id","top_k"}`

> 核验来源：官网 https://agentmemories.ai/competition/（FAQ Q5 学术榜模型约束、赛程与奖金），
> 接口细节对照第三方参赛者整理的公开契约文档（与官网规则一致）。
> 官方 API Guide 是 SPA 页面，`https://agentmemories.ai/api-guide` 当前抓不到完整细节——**Full 前务必重核官方文档**（组织方可能更新操作限制）。

## 2. 架构

```
POST /add ──► 抽取 (gpt-4o-mini / passthrough) ──► 向量化 (text-embedding-v4)
                    │
                    ▼
              冲突裁决 (LLM judge / none) ──► pgvector 持久化 (单事务, 200 前完成)
                    │                              ▲
                    │                    add_requests 幂等表 (request_id 主键, 409 判定)
                    ▼
POST /search ─► 混合检索 ──► RRF 融合 ──► (可选)上下文扩展 ──► (可选)rerank ──► 取 top_k
   (稠密 pgvector 余弦)  (词法 tsvector+ts_rank)        输出 {"data":[...]}，只给证据
```

扩展点都在 `app/pipeline/` 下实现为可替换策略，`config.yaml` 里只做选型：

| 环节 | 默认 | 可选项 | 文件 |
|---|---|---|---|
| 抽取 extraction | `passthrough`（原文直存，零成本） | `llm_extract`（gpt-4o-mini 抽事实/事件/偏好） | `pipeline/extract.py` |
| 冲突消解 conflict | `llm_judge`（无 LLM 时自动降级 `none`） | `none`（全部入库） | `pipeline/conflict.py` |
| 检索 retrieval | `hybrid_rrf`（稠密+词法+RRF） | `dense` / `lexical`（消融基线） | `pipeline/retrieve.py` |
| 精排 rerank | `none` | `cross_encoder` / `llm_judge`（骨架已给，见文件注释） | `pipeline/rerank.py` |

Prompt 模板在 `app/prompts/`（`extract_facts.txt`、`judge_conflict.txt`），直接改文本即生效。

## 3. 快速开始

### 方式 A：docker-compose 一键起（推荐，VPS 同构）

```bash
cd ~/workspace/aml
cp .env.example .env   # 填 MEMORY_API_KEY、EMBEDDING_API_KEY、LLM_API_KEY
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

### 方式 B：本地 venv（无 docker 时联调用）

需要一台 Postgres 16 + pgvector（见 docker-compose 里的 `db` 服务，或云数据库）：

```bash
cd ~/workspace/aml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填 key；DATABASE_URL 指向你的 pg
export CONFIG_PATH=./config.yaml
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 零成本联调模式（先跑通契约，不花一分钱）

`.env` 里设 `EMBEDDING_PROVIDER=stub`，`config.yaml` 里 `extraction.enabled: false`：
embedding 用确定性伪向量，Add 不调任何外部 API。然后跑 Smoke：

```bash
python scripts/smoke.py --base-url http://127.0.0.1:8000 --api-key "$MEMORY_API_KEY"
```

> ⚠️ stub 仅用于本地契约联调。DB 会记录 embedding 身份，stub 和真实模型不能混用同一个库；
> **正式评测必须用 `text-embedding-v4`**，否则成绩无效。

## 4. 用户决策清单（需要你拍板的）

1. **抽取开关** `extraction.enabled`：`false`=原文直存（便宜、召回靠原文）；`true`=gpt-4o-mini 抽事实/事件/偏好（贵，但证据更干净、冲突消解才有意义）。建议：Smoke 用 false，Full 前用公开数据（如 LoCoMo）对比一次再定。
2. **冲突消解** `conflict.strategy`：`llm_judge` 处理更新/矛盾/删除意图（记忆治理 D 项是文本赛道的明确考察点）；`none` 最快最便宜。默认 `llm_judge`。
3. **检索策略** `retrieval.strategy`：默认 `hybrid_rrf`；`dense`/`lexical` 留作消融对比。`seed_k` 建议 ≥ 100（正式 top_k=100）。
4. **Reranker**：默认 `none`。学术榜 reranker 不限，加本地 cross-encoder 是性价比最高的升级点（不耗 LLM 配额）；`llm_judge` 精排效果可能好但 Full 全程成本极高。
5. **上下文窗口** `retrieval.context_window`：给种子补充同会话相邻消息。默认 0（关闭）；开 >0 会改变语义，建议先做对比实验。
6. **VPS 选型**：2c4g 起步（见 `deploy/VPS_DEPLOY.md`）；embedding/LLM 都走远端 API，VPS 本身不跑模型，瓶颈是并发连接数和带宽。
7. **Key 申请**：这是你的动作——在官网 Evaluation 页申请（申请次日 19:00 前发放），**最晚 10/2 发出**，否则赶不上 10/5 的首次 Full。

## 5. Smoke 与压测

```bash
# 契约自检（12 项：health/add 回传/幂等/409/立即可见/top_k 上限/options/空用户/跨用户隔离/无重复）
python scripts/smoke.py --base-url http://127.0.0.1:8000 --api-key "$MEMORY_API_KEY"

# 并发压测（默认 Add 64 / Search 256 并发；Add 压测走真实 embedding/LLM，有成本，先小量试）
python scripts/load_test.py --base-url http://127.0.0.1:8000 --api-key "$MEMORY_API_KEY" \
  --adds 16 --searches 64 --add-concurrency 16 --search-concurrency 32

# 数据库管理
python scripts/admin.py --database-url "$DATABASE_URL" stats
python scripts/admin.py --database-url "$DATABASE_URL" purge-prefix --user-prefix 'smoke:xxx' --confirm
```

## 6. 成本估算（粗估，正式跑之前按实际用量重算）

| 项 | 说明 | 估算 |
|---|---|---|
| Embedding（text-embedding-v4） | 文本赛道 ~300M tokens | 约 ¥210 上限（DashScope 定价很低） |
| LLM 抽取/裁决（gpt-4o-mini） | **主要成本**，取决于 extraction/conflict 开关与消息量 | $0.15 / 1M input tokens，按实际开 |
| VPS + 带宽 | 2c4g 按量/包月，评测约 72 小时 + 准备期 | ¥100–300 |
| 合计 | | **¥500–1500 量级**（不开 LLM 抽取可压到 ¥300 内） |

省成本技巧：Smoke/联调用 `stub`；公开数据回放调参时 `extraction.enabled=false`；
Full 前再开 `true`，且先在小样本上验证 prompt 与成本。

## 7. 部署到公网 VPS

第二期要求**自部署公网 Add/Search API**（平台不再代部署），评测后不要求继续开放。
完整步骤见 [deploy/VPS_DEPLOY.md](deploy/VPS_DEPLOY.md)（含 Docker、Caddy 自动 HTTPS、防火墙、公网验证、评测后下线）。

## 8. 合规与注意事项

- **评测数据只能用于当次运行**，不得用于训练/分析/分享，跑后 30 天内删除（`scripts/admin.py purge-prefix`）。
- **不要**针对公开题目硬编码，不要跨样本共享状态——这是红线，会被复核处理。
- embedding 身份 guard：换模型/维度必须换新库，服务启动时会拒绝混用。
- Add 并发 64 / Search 并发 256：`server.db_pool_max`（默认 50）、uvicorn `WORKERS`（默认 2）可按压测结果调。
- Full 前重核官方 API Guide 与规则（超时、限流等操作限制可能更新）。
- `.env` 里的 key 不要提交到 git（`.gitignore` 已配好请自行确认）。
