# Phase 1 · Enriched Replay：数据地基消融

实验时间：2026-10-03
方法：同一套代码、同一 1527 题，仅 Add 输入不同。
- Baseline：content=纯文本，role 全 user，无 timestamp，session_id=dia_id
- Enriched：content="{speaker}: {text}"，role=speaker_a→user/speaker_b→assistant，
  timestamp=session 日期转 Unix 毫秒，session_id 保持 dia_id（计分口径一致）
- 独立 user 命名空间，不混杂。n=1527 完全对齐，可比。

## 总体

| 指标 | baseline | enriched | 提升 |
|---|---|---|---|
| R@1 | 0.2461 | 0.4554 | **+85%** |
| R@5 | 0.4089 | 0.6534 | +60% |
| R@10 | 0.4666 | 0.7178 | +54% |
| R@20 | 0.5253 | 0.7580 | +44% |
| R@100 | 0.7362 | 0.8661 | +18% |
| MRR | 0.3543 | 0.6164 | +74% |
| nDCG@100 | 0.4210 | 0.6467 | +54% |

## 分题型 R@1

| 题型 | baseline | enriched | 提升 |
|---|---|---|---|
| single_hop | 0.3020 | 0.5395 | +79% |
| temporal | 0.3201 | 0.5638 | +76% |
| multi_hop | 0.0441 | 0.1648 | **+274%** |
| open_domain | 0.0843 | 0.1798 | +113% |

## 结论

1. 数据地基是最大杠杆：仅把 speaker 名喂进 content，R@1 几乎翻倍。验证了评审报告
   "最大问题不是算法，而是数据地基"的判断。
2. multi_hop 提升 3.7 倍：speaker 名让"X 说了什么"类问题可匹配。
3. temporal +76%（注意：timestamp 已发送但生产未使用，纯 speaker 前缀的效果；
   日期增强的增量待测）。
4. 本实验是"乐观上界"（假设平台真实评测会发送 speaker/timestamp）；
   baseline 是"悲观下界"。真相介于之间。

## 待测

- 生产端日期增强（timestamp→content 拼日期）的增量
- lexical english 切换后与 enriched 数据的组合效果
