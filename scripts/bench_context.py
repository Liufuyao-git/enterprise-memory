#!/usr/bin/env python3
"""
bench_context.py — 上下文扩展编码对比实验
对比 window=0/1/2 在样本0上的效果
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from eval_locomo import run_evaluation, load_dataset, DATA_PATH

data = load_dataset(DATA_PATH)
sample0 = [data[0]]  # 只跑样本0

for window in [0, 1, 2]:
    print(f"\n{'='*50}")
    print(f"window={window}")
    print('='*50)
    summary = run_evaluation(
        data=sample0,
        method="all",
        top_ks=[1, 3, 5],
        cache_dir=Path("tests/vector_cache"),
        context_window=window,
    )
    for method, res in summary["methods"].items():
        r = res["recall"]
        print(f"{method:8s}: R@1={r['R@1']:.3f} R@3={r['R@3']:.3f} R@5={r['R@5']:.3f} MRR={res['mrr']:.3f}")
