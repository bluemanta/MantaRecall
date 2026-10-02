#!/usr/bin/env python3
"""从 eval/reports/replay_full*.json 抽取看板数据 -> eval/dashboard_data.json.

每轮实验的"做了什么"维护在 NOTES 里；新增实验轮次时在这里加一条。
"""
import json, glob, os, re
from datetime import datetime

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_data.json')

# run_id 关键字 -> (展示名, 做了什么)
NOTES = {
    'replay_full_20261001T145955': (
        '基线',
        '首次全量 LoCoMo 回放基线：hybrid_rrf（dense 向量 + lexical 关键词双路，RRF 融合），无 reranker。',
    ),
    'replay_full_reranker_20261001T165046': (
        'Reranker',
        'RRF 之后加本地 cross-encoder 重排（ms-marco-MiniLM-L-6-v2，top_n=50，baked 进镜像，torch 单线程）。',
    ),
}

ABLATION_NOTES = {
    'reranker2': ('Reranker（修复后）', 'lexical 改 OR 语义后，生产配置全量回放：hybrid_rrf + 本地 cross-encoder reranker（top_n=50）。'),
    'ablation2_dense': ('Dense（修复后）', 'lexical 改 OR 语义后重跑：只走 dense 向量检索一路，reranker 关闭。'),
    'ablation2_lexical': ('Lexical（修复后）', 'lexical 改 OR 语义后重跑：只走 lexical 关键词检索一路，reranker 关闭。'),
    'ablation2_hybrid': ('Hybrid（修复后）', 'lexical 改 OR 语义后重跑：hybrid_rrf 双路融合，reranker 关闭，验证融合是否吃到两路优点。'),
    'ablation_dense': ('消融·Dense', '只走 dense 向量检索一路，reranker 关闭：验证纯语义检索的召回上限。'),
    'ablation_lexical': ('消融·Lexical', '只走 lexical 关键词检索一路，reranker 关闭：验证纯关键词检索的召回上限。'),
    'ablation_hybrid': ('消融·Hybrid', 'hybrid_rrf 双路融合，reranker 关闭：与单路对照，看融合是否吃到两路优点。'),
}

# 消融报告里 config.retrieval 是脚本静态标注（不可靠），以文件名为准
ABLATION_RETRIEVAL = {
    'reranker2': 'hybrid_rrf',
    'ablation2_dense': 'dense-only',
    'ablation2_lexical': 'lexical-only',
    'ablation2_hybrid': 'hybrid_rrf',
    'ablation_dense': 'dense-only',
    'ablation_lexical': 'lexical-only',
    'ablation_hybrid': 'hybrid_rrf',
}

# 带 reranker 的生产全量轮次（reranker 状态按文件名单独判定）
RERANKER_ON_RIDS = ('reranker2',)

# PRF 验证轮：客户端两轮检索 + RRF 融合，verdict=否
PRF_NOTES = {
    'replay_prf_norank': ('PRF（验证·否）', '客户端两轮 PRF：首轮 top5 文本扩展 query 再查，两轮 RRF 融合。multi_hop R@100 +5.3pp 但 query 漂移拖累 single_hop/temporal，总体 R@1 -15%，verdict：不值得服务端实现。'),
}

# 第一轮消融（lexical 修复前）的旧报告已被第二轮取代，不进看板
SKIP_RIDS = ('replay_ablation_dense_norank', 'replay_ablation_lexical_norank')

OVERALL_KEYS = ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@100', 'mrr', 'ndcg@100']
CAT_LABEL = {'single_hop': '单跳', 'temporal': '时间', 'multi_hop': '多跳', 'open_domain': '开放域'}


def run_id_of(path):
    base = os.path.basename(path)
    return base[:-5] if base.endswith('.json') else base


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(REPORTS_DIR, 'replay_*.json'))):
        rid = run_id_of(path)
        # 小样本验证报告不进看板；第一轮消融旧报告已被第二轮取代
        if 'conv-30' in rid or any(rid.startswith(s) for s in SKIP_RIDS):
            continue
        d = json.load(open(path))
        cfg = d.get('config', {})
        ev = d.get('eval', {})
        overall = {k: ev.get('overall', {}).get(k) for k in OVERALL_KEYS}
        per_cat = {}
        for cat, vals in ev.get('per_category', {}).items():
            if cat not in CAT_LABEL:
                continue
            per_cat[cat] = {k: vals.get(k) for k in OVERALL_KEYS}
            per_cat[cat]['n'] = vals.get('n_questions')
        # 名字与备注
        label, notes = NOTES.get(rid, (None, ''))
        if label is None:
            for key, (lab, note) in ABLATION_NOTES.items():
                if key in rid:
                    label, notes = lab, note
                    break
        if label is None:
            for key, (lab, note) in PRF_NOTES.items():
                if rid.startswith(key):
                    label, notes = lab, note
                    break
        if label is None:
            label, notes = rid, ''
        # 配置摘要（消融报告的 config 标注不可靠，以文件名为准）
        retrieval = str(cfg.get('retrieval', ''))[:40]
        reranker_on = None
        for key, ret in ABLATION_RETRIEVAL.items():
            if key in rid:
                retrieval = ret
                reranker_on = any(k in rid for k in RERANKER_ON_RIDS)
                break
        if reranker_on is None and rid.startswith('replay_prf_norank'):
            retrieval = 'hybrid_rrf+PRF'
            reranker_on = False
        if reranker_on is None:
            reranker_on = bool(cfg.get('reranker', {}).get('enabled')) if isinstance(cfg.get('reranker'), dict) else ('rerank' in rid)
        runs.append({
            'id': rid,
            'label': label,
            'time': d.get('generated_at', ''),
            'notes': notes,
            'retrieval': retrieval or rid,
            'reranker': '开启' if reranker_on else '关闭',
            'overall': overall,
            'per_category': per_cat,
            'timing': d.get('timing', {}),
        })
    # 按时间排序（文件名排序会把 ablation 排到前面）
    runs.sort(key=lambda r: r['time'])
    # 基线 delta 参照：第一轮
    out = {'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'), 'runs': runs,
           'metric_labels': {'recall@1': 'R@1', 'recall@5': 'R@5', 'recall@10': 'R@10',
                             'recall@20': 'R@20', 'recall@100': 'R@100',
                             'mrr': 'MRR', 'ndcg@100': 'nDCG@100'}}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f'{len(runs)} runs -> {OUT}')
    for r in runs:
        print(' -', r['label'], r['overall'].get('recall@1'), r['overall'].get('mrr'))


if __name__ == '__main__':
    main()
