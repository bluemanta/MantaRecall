# Phase 1 总结报告（2026-10-03）

范围：数据地基、lexical 消融、HNSW 对照、生产落地决策。
结论先行：**三个变量全部进入生产，无回退**。当前生产镜像 `aml-api:phase1-20261003` 已包含全部变更。

## 1. 数据地基盘点

- LoCoMo 原始数据包含 `speaker_a/b`、`session_N_date_time`、每 turn 的 `speaker`。
- 旧 replay 只喂纯文本（role 全 user、无 timestamp、session_id=dia_id），丢掉了 speaker 名、角色、真实 session 分组和日期。
- 生产 passthrough 会再拼 role 前缀，旧 baseline 实际存储近似 `user: <text>`。

## 2. 实验结果

### 2.1 speaker/role/timestamp 组合包（enriched replay，n=1527）

| 指标 | baseline | enriched | 变化 |
|---|---:|---:|---:|
| R@1 | 0.2461 | 0.4554 | **+85.0%** |
| R@5 | 0.4089 | 0.6534 | +59.8% |
| R@10 | 0.4666 | 0.7178 | +53.8% |
| MRR | 0.3543 | 0.6164 | +74.0% |

R@1 分题型：single_hop 0.302→0.540；temporal 0.320→0.564；
multi_hop 0.044→0.165（+274%）；open_domain 0.084→0.180。

注意：这是组合包（speaker 前缀 + role 映射 + timestamp），未做单变量拆分。
生产已按此格式存储（`{role}: {speaker}: {text}`）。

### 2.2 日期增强（生产已部署，conv-48 聚焦验证 n=191）

生产 `PassthroughExtraction`：timestamp 存在则 content 拼 ` [08 May 2023]`，
无则降级为原格式。实测 `user [08 May 2023]: Caroline: ...` 正常。

| 指标（temporal, n=42） | 无日期 | 有日期 | 变化 |
|---|---:|---:|---:|
| R@1 | 0.6151 | 0.6389 | +3.9% |
| R@5 | 0.7579 | 0.7738 | +2.1% |
| R@10 | 0.8254 | 0.8651 | +4.8% |
| MRR | 0.6981 | 0.7286 | +4.4% |

整体（n=191）：R@1 0.4996→0.4933，MRR 0.6867→0.6840，基本持平无回退。

真实价值可能在 Answer 阶段（证据带日期 → "yesterday" 可消解），
retrieval 指标捕捉不到。零风险，保留。

### 2.3 lexical `simple` vs `english`（离线 TEMP 表，n=1446，lexical-only）

| 指标 | simple | english | 变化 |
|---|---:|---:|---:|
| R@1 | 0.0897 | 0.2099 | **+134%** |
| R@5 | 0.2160 | 0.3851 | +78% |
| MRR | 0.1751 | 0.3259 | +86% |

分题型 R@1 全面提升：temporal 0.076→0.233（+207%）最显著。
**已迁移生产**：重建 `content_tsv` 生成列为 `to_tsvector('english', ...)`，
查询改为 `websearch_to_tsquery('english', ...)`，重建 GIN 索引。
词干实测（"running" 命中 "run"）正常，本地 smoke 20/20。

### 2.4 HNSW 精确对照（60 随机向量）

recall@1/10/50/100 全部 = 1.0000。当前数据规模下 HNSW 零召回损失，
无需调参。pgvector 较老不识别 `hnsw.ef_search`，非阻塞项。

## 3. 生产落地决策

| 变量 | 决策 | 状态 |
|---|---|---|
| speaker 前缀 + role 映射 | 进 | 已上线（`{role}: {speaker}: {text}`） |
| 日期增强（timestamp→日期） | 进 | 已上线，优雅降级 |
| lexical english | 进 | 已上线（DB+代码+索引） |
| HNSW 调参 | 不做 | 无损失 |

当前镜像：`aml-api:phase1-20261003`（= latest）。
回滚点：`aml-api:phase0-20261003-fix2`（日期前）、
`aml-api:pre-phase0-20261003`（Phase 0 前）。

## 4. 风险与回滚

- 日期增强：纯条件逻辑，无 timestamp 时行为与旧版完全一致。
- english 迁移：不可逆的 DB 列重建（已执行）；回滚需重建列回 simple
  并重部署旧镜像。已验证 smoke 20/20。
- 回滚命令：`docker tag aml-api:phase0-20261003-fix2 aml-api:latest && docker restart aml-api-1`
 （注意：回滚镜像的 lexical 为 simple，需同步重建 tsvector 列）。

## 5. 待办（Phase 1 收尾）

- [ ] commit + push（用户已说"后面再说"，保持阻塞）
- [ ] 评估数据 30 天内删除（eval:locomo* 命名空间，共 12445 行）
- [ ] 下一步：Phase 2（multi_hop query 分解）或直接准备 Full

---
*报告生成：2026-10-03。实验 JSON 见 eval/reports/。*
