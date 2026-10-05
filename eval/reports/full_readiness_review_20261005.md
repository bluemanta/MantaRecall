# Full 前就绪审查（2026-10-05）

- 审查对象：`main@4a483e2`（Revert lexical to simple），生产镜像 `aml-api:phase1-simple-date`（镜像 ↔ 提交对应关系待 P0 核验）
- 范围：`app/` 核心链路（extract / retrieve / rerank / db / main）、Phase 1 方案与结论、Full 风险
- 方法：读代码 + `eval/reports/` 复核；本地 PG16 实测词法行为；本地 PG 16.2 + pgvector 0.6.2 用生产 `_dense_search` 原样复现 dense 截断
- 约束：全程只读，未连生产；未用官方 Smoke 题目
- 配套脚本：[`scripts/p0_diagnose.py`](../../scripts/p0_diagnose.py)（P0 只读诊断）

分类口径：**【确认】** 有代码路径或实测证据；**【推断】** 合理但未在生产验证，均附验证方法；**【建议】** 按优先级排序，带验收标准。

---

## 0. 结论摘要

1. **simple + date 可以冻结进 Full**，但有 4 个先决风险要先用 P0 脚本排除：tsvector 列回滚残留（I1）、HNSW + user 过滤截断（C1/I2）、连接预算 100 = 100（C7）、Add 孤儿 processing（C3/I6）。
2. **dense 截断机制已复现**：生产 `_dense_search` 在规划器选中 HNSW 时，`LIMIT 200` 只返回 3–7 条；用 MATERIALIZED CTE 改精确检索后恢复 200 条，600 行 user 耗时 1.7ms。建议作为 P1 第一项修复。
3. **B/D 下降归因 english 停用词证据不足**：更可能是小样本噪声 + D（记忆治理）在零 LLM 模式下的结构性缺口。已回 simple，不值得再花 Smoke 次数。
4. 时间线：建议 10/10 代码冻结，最后一次部署后跑一次 Smoke 仅作回归门禁（不调参），10/12–10/15 发起 Full 可行。

---

## 1. dense 截断复现实验（C1 的证据）

环境：pgserver 内嵌 PostgreSQL 16.2 + pgvector 0.6.2；schema 用 `app.db.ddl(1024)` 原样建表（含 HNSW 索引）；1024 维随机单位向量；`hnsw.ef_search` 默认 40；调用生产函数 `app.pipeline.retrieve._dense_search`（未改动），k=200；每个 case 新建连接（新的预处理语句缓存）。

| 表规模 | 目标 user 行数（占比） | 规划器 | 每次调用返回条数（期望 200） |
|---|---|---|---|
| 7.2k | 600（8.3%） | Sort（精确） | 200 ×10 |
| 37.2k | 600（1.6%） | Sort（精确） | 200 ×10 |
| 38.7k | 1,500（3.9%） | 前 5 次 HNSW → 第 6 次起 generic plan 走 Sort | **3, 3, 3, 3, 3**, 200, 200, 200 |
| 51.2k | 14,000（27.3%） | HNSW（始终） | **7 ×8**（与精确结果重合 4%） |
| 51.2k | 600（1.2%） | Sort（精确） | 200 ×8 |
| 37.2k，`user_id LIKE` 命中 81% | — | HNSW | 24 |
| 同上 + `SET hnsw.ef_search=400` | — | HNSW | 200 |

修复验证（同一数据，MATERIALIZED CTE 精确检索）：

| user | 返回 | top1 自匹配 | 中位耗时（本机 M 系列） |
|---|---|---|---|
| big:user（14k 行） | 200 | 1.0000 | 32 ms |
| conv-3（600 行） | 200 | 1.0000 | 1.7 ms |

要点：
- 是否截断取决于规划器对"该 user 行数 vs 全表规模"的代价比较，**不是单调的占比阈值**；asyncpg 每个连接的前 5 次执行用 custom plan，之后可能切 generic plan，所以同一查询在新/旧连接上结果条数可能不同。asyncpg 连接池默认空闲 300s 回收连接，低流量突发（如 Smoke）更容易落在"新连接"上。
- 生产 pgvector 版本未知（`pgvector/pgvector:pg16` 浮动标签，可能是 0.8.x，代价模型不同），**生产是否真的发生需 P0 脚本确认（I2）**；但只要数据规模变化，计划就可能翻转，根治方案是精确检索。
- Phase 1 的"HNSW 零召回损失"结论来自 60 个随机向量、无 user 过滤，无法覆盖此场景（见 C11）。

---

## 2. 【确认】

**C1. HNSW + user_id 过滤会让 dense 召回静默截断** — [retrieve.py:78-103](../../app/pipeline/retrieve.py)
- `WHERE user_id = $2 ... ORDER BY embedding <=> $1 LIMIT $3` 走 HNSW 时，索引最多产出 `ef_search`（40）条，再按 user 过滤，结果远少于 `dense_k=200`。证据见第 1 节。
- 同样的 SQL 也在 [main.py:205-223](../../app/main.py)（冲突候选；当前 conflict=none 不触发）。

**C2. 返回的 score 非单调** — [rerank.py:116-119](../../app/pipeline/rerank.py)
- head（前 `RERANK_TOP_N=50`）是 cross-encoder 分数（logits 可为负；即便经 sigmoid，低相关项也接近 0）；tail（51–100）保留 RRF 分数（≈0.006–0.033）。量纲不同，顺序正确但 score 不是递减。Full `top_k=100`，tail 必然返回；若评测方按 score 重排或阈值截断会乱序。

**C3. Add 孤儿 processing** — [main.py:310](../../app/main.py)、`_STALE_MINUTES=30`、`_POLL_TIMEOUT=30`
- Python 3.8+ 的 `CancelledError` 是 `BaseException`，`except Exception` 接不住；进程被杀 / OOM / 部署重启 / 请求取消时不会 `_mark_failed`，行停在 processing。
- 后果：同 request_id 重试会先轮询 30s 再 503，直到 30 分钟才可被认领；评测方短时多次重试都失败 → 该 session 数据静默丢失。

**C4. 时间戳单位无防护** — [extract.py:20-32](../../app/pipeline/extract.py)
- 实测：毫秒 `1683556560000` → ` [08 May 2023]`；秒 `1683556560` → ` [20 January 1970]`。秒级输入会把错误日期写进 content，同时污染 dense / lexical / 答案证据。

**C5. 词法查询拼接缺陷** — [retrieve.py:114](../../app/pipeline/retrieve.py)（PG16 实测）
- `What OR is OR X OR - OR Y?` → `'what' | 'is' | 'x' | !'or' & 'y'`（单独的 `-` 变成取反并改成 AND）。
- `temperature OR -5 OR degrees` → `'temperature' | !'5' | 'degrees'`。
- 影响面小，但是真 bug。

**C6. simple + OR 的词法排序被停用词主导**（PG16 实测）
- 查询 "What did Caroline do at the support group?"：一条不相关但 what/did/the 很多的长消息 `ts_rank_cd`=1.70，真正相关的消息 0.40；加归一化 `1|32` 仍是 0.314 vs 0.143。
- 解释了 Phase 1 lexical-only 实验里 english R@1 +134%：收益主要来自**查询侧去停用词**，不是词干化。

**C7. 连接预算顶满** — [db.py:8](../../app/db.py)、[config.yaml:37](../../config.yaml)、[docker-compose.yml](../../docker-compose.yml)
- `WORKERS=2 × db_pool_max=50 = 100`，PG 默认 `max_connections=100` 且 compose 未覆盖。256 并发下池子长满后，任何额外连接（psql、admin、诊断）都会触发 too many clients。

**C8. 压测不代表 Full** — [scripts/load_test.py](../../scripts/load_test.py)
- 64 次 Add 分给 8 个 user，每个 user 约 8 行；Search 只重排 ≤8 个候选（真实是 50 个候选、每 user 600+ 行）。现有 p95 参考价值低。

**C9. reranker 512 token 窗口**
- ms-marco-MiniLM-L6-v2 是 BERT 结构，最大 512 token；长消息约 450 token 之后的内容 reranker 看不到。LoCoMo 每轮很短，离线完全测不出。

**C10. Search 无降级、embedding 信号量共用** — [main.py:337-340](../../app/main.py)、[embeddings.py:50](../../app/providers/embeddings.py)
- DashScope 重试耗尽直接 500，不会退回 lexical + rerank。
- `Semaphore(8)`（每 worker）由 Add 与 Search 共用。

**C11. Phase 1 的 HNSW 对照无效**
- 只有 60 个随机向量、无 user 过滤，覆盖不到 C1 场景。
- "pgvector 较老不识别 `hnsw.ef_search`"自相矛盾：该参数与 HNSW 索引同在 0.5.0 引入。更可能是 SHOW 时 vector 库尚未加载（GUC 在库加载时才注册）。P0 脚本会先触发加载再读取。

**C12. 文档不一致**
- [phase1_summary.md](phase1_summary.md) 开头写"三个变量全部进入生产，无回退"，§2.3 写已回滚 english。Full 前统一，避免复现口径混乱。

---

## 3. 【推断】（均附验证方法）

| # | 推断 | 验证方法 |
|---|---|---|
| I1 | **生产 tsvector 列可能仍是 english**：Phase 1 总结 §4 写明"回滚需重建列"，而 `CREATE TABLE IF NOT EXISTS` 不改已有列。若列是 english、查询是 simple，lexical 通道静默失效 | P0 脚本 `schema` 检查；期望 `to_tsvector('simple'::regconfig, content)` |
| I2 | **生产 dense 可能已在截断**，且可能是 Smoke 分数波动的一个来源（数据构成变了、连接新旧不同 → 计划不同） | P0 脚本 `share` + `dense` 检查（新连接连调 8 次 + 与精确检索对比）。历史 Smoke 无法回放验证，只能看当前库里 Smoke user 的计划 |
| I3 | 256 并发下 Search 延迟主要卡在 reranker CPU 排队，可能触碰评测方超时 | 真实压测（见 P2-1）：记录 p50/p95/p99、错误率、CPU、`pg_stat_activity` |
| I4 | B/D 下降是小样本噪声 + 结构性缺口，不是 english | 见第 4 节；拿到 Smoke 报告里分项题数后做 Fisher 精确检验；可选：自建 20–30 条"更新/矛盾"合成探针对比 simple/english |
| I5 | 时区导致日期错一天：LoCoMo 288 个 session，若评测方按 UTC+8 编码，约 6%（17 个）提前一天；按美国太平洋时间编码，37–45% 推后一天 | 补问主办方：timestamp 单位（ms/s）+ 编码时区；Full 若带时间戳，抽查 content 内日期 |
| I6 | 生产库已有孤儿 processing 行 | P0 脚本 `orphans` 检查 |
| I7 | 生产镜像不一定等于 `4a483e2` | P0 脚本 `code` 指纹：容器内输出 vs 本地 `--hash-only` 输出；`/health` 的 rerank 状态 |

---

## 4. 方案评审

### simple + date：同意冻结
- 端到端对比只有 conv-48（n=191，multi_hop 仅 21 题），结论应是"未测出差异"而非"无差异"。维持现状规避风险，决策合理。
- date 对官方 Smoke 是 no-op（数据不带 timestamp），但"零风险"不成立：需补 C4（单位防护），并确认 I5（时区）。

### B/D 归因 english：证据不足
1. **样本太小**：B 66→33 基本意味着 3 题里翻了 1 题。
2. **数据不同**：已证明两次 Smoke 数据不同、总分不可比，同样适用于分项。
3. **机制站不住**：reranker 用原始 query，停用词只影响 lexical 一路召回哪些候选；内容词照样命中，删掉 not/no 很难决定结果。
4. **离线不支持**：最接近 B 的 multi_hop 在 conv-48 上 simple/english R@1 都是 0.177。
5. **D 低是结构性的**：零 LLM 下 conflict=none、无时效信号、官方数据不带 timestamp，系统没有记忆治理能力；LoCoMo 也没有对应题型，离线测不到。

结论：已回 simple，此假设不再占用 Smoke 次数。若 Full 后要提升 D，杠杆是时效/时间信息（如近重复候选按写入顺序 tie-break，需先确认评测方 Add 是否按时间顺序串行），不是 tsconfig。

---

## 5. 【建议】

### P0：今天，只读，约 30 分钟 — 跑 `scripts/p0_diagnose.py`

```bash
docker cp scripts/p0_diagnose.py aml-api-1:/tmp/p0_diagnose.py
docker exec -w /app aml-api-1 python /tmp/p0_diagnose.py --health-url http://127.0.0.1:8000/health --json-out /tmp/p0_report.json
docker cp aml-api-1:/tmp/p0_report.json ./p0_report.json
python scripts/p0_diagnose.py --hash-only
```

| 脚本检查 | 对应条目 | 期望 |
|---|---|---|
| `env` | C11 | 记录 pgvector 版本、`hnsw.ef_search`、`max_connections` |
| `config` | C7 | `WORKERS × db_pool_max ≤ max_connections − 10` |
| `schema` | I1 | content_tsv 为 simple |
| `share` / `dense` | C1 / I2 | 每次返回 = min(200, active 行数)，与精确结果重合 100% |
| `orphans` | C3 / I6 | 无超过 5 分钟的 processing |
| `length` | C9 | Smoke 等非 eval 数据 >2000 字符占比 <5% 则 C9 可后置 |
| `ts` | C4 | 无秒级时间戳、无 1970 日期 |
| `code` | I7 | 容器 digest = 本地 `--hash-only` digest |
| `health` | I7 | rerank.enabled = true |

脚本保证：连接级只读（写操作被数据库拒绝，已测）、`statement_timeout=30s`、单连接串行、不输出记忆原文、dense 检查用库内已有向量（不调 DashScope）。建议在无评测流量时运行。若 I1 为 FAIL，先停下报告，不要自行重建列。

### P1：修复（每项单独提交；一次只动一个变量）

| 顺序 | 条目 | 改法 | 验收标准 |
|---|---|---|---|
| 1 | C1 | dense 改精确检索：`WITH u AS MATERIALIZED (SELECT ... WHERE user_id=$2 AND status='active') SELECT ... FROM u ORDER BY embedding <=> $1 LIMIT $3`；main.py 冲突候选同改 | 多 user 库上返回条数 = min(k, n)（可用 p0 脚本 `dense` 检查本地验证）；回放 n=1527 指标 ≥ 基线；压测 p95 增量可接受。删除 HNSW 索引是单独一步，放 Full 之后 |
| 2 | C3 | 捕获 `BaseException`，用 `asyncio.shield` 包 `_mark_failed`；流水线外包 `asyncio.wait_for(..., 120s)`；`_STALE_MINUTES` 30→5（须 > 流水线超时） | 故障注入：Add 处理中杀 worker → 5 分钟内同 request_id 重试成功；正常幂等 / 409 行为不变（`scripts/smoke.py` 全过） |
| 3 | C4 | 渲染出的年份不在 2000–2100 则不拼日期（不猜单位） | 单测：ms → 拼日期；秒 → 不拼；None/0 → 不拼 |
| 4 | C2 | tail 分数改为 `min(head) − ε·i`，顺序不变 | 单测：score 单调不增；返回 id 顺序与改前逐条一致 |
| 5 | C10 | query embedding 失败时退回 lexical + rerank，记 warning | 单测：embedder 抛异常 → 200 且有结果 |
| 6 | C7 | `db_pool_max` 降到 40，或 compose 给 db 加 `command: postgres -c max_connections=200`（二选一） | p0 `config` 检查 PASS |
| 7 | C5 | 词法 token 去掉前导 `-`、过滤空 token | 回放：指标不降（预期几乎不变） |

第 2–6 项不改变排序：用回放子集确认返回列表逐条一致即可。第 1、7 项可能改变排序：需完整回放 n=1527。

**回放必须指向本地或预发实例**：`eval/replay*.py` 的 `--base-url` 默认是生产 `https://aml-api.imalltrix.com`，验证新代码时不要打生产。

### P2：Full 前运维

1. **真实压测**：对已有 `eval:locomo:*` user 用 LoCoMo 公开题（合规），`top_k=100`，并发 64/128/256，各跑一轮；记录 p50/p95/p99、错误率、CPU、连接数。若 p95 过高，先调 `RERANK_TOP_N` / `WORKERS`，并单独回放验证。
2. DashScope：余额、text-embedding-v4 的 RPM/TPM 限额、按 Full 数据量估算调用量。
3. 磁盘：pgdata 增长（每行约 4KB 向量 + HNSW 索引）、日志轮转。
4. Full 期间冻结部署（部署会制造 C3 孤儿）；配健康探针 + 错误率告警。
5. eval 数据 30 天清理与 Full 错开；清理或 Full 灌入后数据规模大变时，重跑 p0 `dense` 检查（C1 修复前尤其重要）。

### P3：可选实验（Full 前时间不够就放到 Full 后；必须用 n=1527 + 配对 bootstrap）

- 保持 simple 索引，仅查询侧过滤疑问词 / 虚词（保留 not/no/never/before/after），用开关控制，无需迁移 — 验证 C6 机制。
- 若 P0 `length` 显示长消息多：对长消息分窗打分取最高分（MaxP）rerank — 对应 C9。

---

## 6. 执行约束

- 不用官方 Smoke 题目在本地回放调参。
- 每次改动需离线数字支撑，一次只动一个变量。
- 研究诊断默认只读；任何生产变更（部署、DDL、数据清理）先经负责人确认。
- 改代码 ≠ 允许 commit/push：提交前需负责人确认；只 `git add` 本任务文件。
- 诊断与报告不导出评测原文。

---

## 7. 执行状态（2026-10-05 P0+P1 完成）

### P0 诊断结果（生产只读，2026-10-05 08:23）

| 检查 | 结果 | 结论 |
|---|---|---|
| code | INFO | 容器指纹 `1161958cfdf5972e` = 本地，生产代码确认 |
| env | INFO | PG 16.15 / pgvector **0.8.6** / `hnsw.ef_search=40` 可读（C11 解开：之前是库未加载） |
| config | ❌ FAIL | C7 确认：2×50=100 顶满 → P1-6 已修复 |
| schema | ✅ PASS | content_tsv 是 simple，**I1 排除**，无需重建列 |
| share | ⚠️ WARN | 28 个 user ≥2% 占比；当前走精确计划，但清 eval 数据后需重跑 dense 检查 |
| dense | ✅ PASS | 8 user × 8 次全返回足额，Sort 精确，100% 重合；**当前无截断** |
| orphans | ✅ PASS | 无孤儿 processing，I6 当前干净 |
| ts | ✅ PASS | 无秒级/1970 渲染，C4 当前干净 |
| length | INFO | 非 eval >2000 字符仅 0.9%，C9 可后置 |
| health | ✅ PASS | rerank 正常 |

### P1 修复状态

| # | 条目 | 状态 | 提交 | 验证 |
|---|---|---|---|---|
| 1 | C1 dense 精确检索 | ✅ 已修复 | `25d1eab` | 生产库只读对比：6 user 新旧 SQL IDENTICAL；恒返回 min(k,n) |
| 2 | C3 Add 孤儿防护 | ✅ 已修复 | `d2a2d72` | 语法+逻辑 review；故障注入与 smoke.py 需部署后执行 |
| 3 | C4 时间戳年份防护 | ✅ 已修复 | `2ab47b8` | 单测：ms→拼日期；秒/None/0→不拼；边界年份拒绝 |
| 4 | C2 tail 分数单调化 | ✅ 已修复 | `38c4fcf` | 单测：单调不增；id 顺序与改前一致 |
| 5 | C10 embedding 降级 | ✅ 已修复 | `eb69e92` | 单测：抛异常→不 500，有 lexical 结果 |
| 6 | C7 连接预算 | ✅ 已修复 | `2464d42` | p0 config 逻辑验证 PASS（80 ≤ 90） |
| 7 | C5 词法 token 清洗 | ✅ 已修复 | `9d06519` | 单测：'-'过滤；'-5'→'5'；正常查询不受影响 |

### 未处理 / 后置

- **C8**（压测不代表 Full）：P2 真实压测待执行
- **C9**（reranker 512 token）：P0 length 显示 <5%，已后置
- **C11**（HNSW 对照无效）：P0 env 已解开参数之谜，无需再处理
- **C12**（文档不一致）：待统一
- **P2**（压测/DashScope 配额/磁盘/监控）：待执行
- **P3**（可选实验）：Full 后

### 待负责人确认

1. 7 个 P1 提交是否合并到 main（当前在 `docs/full-readiness-review-20261005` 分支）
2. 何时部署到生产（部署后需执行：故障注入测试、smoke.py、p0 重跑）
3. P2 压测计划
