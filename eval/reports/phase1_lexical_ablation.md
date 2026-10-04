# Phase 1 · Lexical 消融：simple vs english

实验时间：2026-10-03
方法：服务器 Postgres 只读实验。TEMP 表复刻生产 memories 表语料（eval:locomo:*，5513 行），
同一批 1446 个 QA，同一查询构造（词 OR 连接 + websearch_to_tsquery），仅 tsconfig 不同。
计分：recall@k / MRR，lexical-only。

## 总体（n=1446）

| 指标 | simple | english | 提升 |
|---|---|---|---|
| R@1 | 0.0897 | 0.2099 | **+134%** |
| R@5 | 0.2160 | 0.3851 | +78% |
| R@10 | 0.2930 | 0.4585 | +56% |
| R@20 | 0.3726 | 0.5265 | +41% |
| R@100 | 0.6197 | 0.6966 | +12% |
| MRR | 0.1751 | 0.3259 | **+86%** |

## 分题型 R@1（标签已按 LoCoMo 官方映射修正：1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop）

| 题型 | simple | english |
|---|---|---|
| multi_hop (n=267) | 0.0221 | 0.0424 |
| temporal (n=294) | 0.0760 | 0.2330 |
| open_domain (n=89) | 0.0337 | 0.0955 |
| single_hop (n=796) | 0.1237 | 0.2703 |

## 结论

english 配置（词干 + 停用词）全面碾压 simple，尤其头部排名（R@1 翻倍多）。
temporal 类提升最大（3 倍），与 smoke 弱项对齐。

## 落地事项

1. DB 迁移：content_tsv 生成列从 to_tsvector('simple', ...) 改为 to_tsvector('english', ...)（需 DROP+重建，5k 行秒级）
2. 代码：retrieve.py 的 websearch_to_tsquery('simple', ...) 改为 'english'；db.py schema 同步
3. 等 enriched replay 跑完后实施，避免重写表锁阻塞 replay 写入
