#!/usr/bin/env bash
# 稳健全量评测：每个样本独立子进程，超时重试，最终合并结果
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
cd "$PROJ"

CACHE_DIR="tests/vector_cache_sp"
OUT_DIR="tests/sample_results"
FINAL="tests/eval_report_final.json"
TIMEOUT=300   # 每个样本最多5分钟
RETRIES=3

mkdir -p "$OUT_DIR"

SAMPLE_IDS=(0 1 2 3 4 5 6 7 8 9)

for i in "${SAMPLE_IDS[@]}"; do
    out_file="$OUT_DIR/sample_${i}.json"
    if [ -f "$out_file" ]; then
        echo "[sample $i] ✅ 已有缓存结果，跳过"
        continue
    fi

    for attempt in $(seq 1 $RETRIES); do
        echo "[sample $i] 第${attempt}次尝试..."
        if timeout $TIMEOUT python3 scripts/eval_locomo.py \
            --method all \
            --sample-id "$i" \
            --window 1 \
            --iterative \
            --cache-dir "$CACHE_DIR" \
            --output "$out_file" \
            2>&1 | tail -5; then
            echo "[sample $i] ✅ 完成"
            break
        else
            echo "[sample $i] ⚠️  第${attempt}次失败，等3秒重试..."
            sleep 3
            rm -f "$out_file"
        fi
    done

    if [ ! -f "$out_file" ]; then
        echo "[sample $i] ❌ 3次均失败，跳过"
    fi
done

echo ""
echo "=== 合并结果 ==="
python3 - <<'EOF'
import json, os, glob, numpy as np
from collections import defaultdict

results_dir = "tests/sample_results"
files = sorted(glob.glob(f"{results_dir}/sample_*.json"))
print(f"找到 {len(files)} 个样本结果")

# 合并各样本结果
merged = defaultdict(lambda: {
    "recall": defaultdict(list),
    "mrr": [],
    "by_category": defaultdict(lambda: defaultdict(list)),
})
total_qa = 0
skipped_qa = 0
top_ks = None

for f in files:
    with open(f) as fp:
        d = json.load(fp)
    total_qa += d["total_qa"]
    skipped_qa += d["skipped_qa"]
    top_ks = d["top_ks"]
    for method, mv in d["methods"].items():
        for k in top_ks:
            key = f"R@{k}"
            n = len(d.get("raw", {}).get(method, {}).get(f"recall_{k}", []))
            # 加权平均（用 total_qa 权重）
            merged[method]["recall"][k].append((mv["recall"][key], d["total_qa"]))
        merged[method]["mrr"].append((mv["mrr"], d["total_qa"]))
        for cat_str, cv in mv.get("by_category", {}).items():
            for k in top_ks:
                merged[method]["by_category"][cat_str][k].append(
                    (cv["recall"][f"R@{k}"], cv["n"])
                )

def wavg(pairs):
    """加权平均"""
    total_w = sum(w for _, w in pairs)
    if total_w == 0: return 0.0
    return sum(v * w for v, w in pairs) / total_w

CATEGORY_NAMES = {
    "1": "单跳事实 (Single-hop Fact)",
    "2": "时间推理 (Temporal)",
    "3": "多跳推理 (Multi-hop)",
    "4": "开放域 (Open-ended)",
}

summary = {"total_qa": total_qa, "skipped_qa": skipped_qa, "top_ks": top_ks, "methods": {}}
for method, res in merged.items():
    m_sum = {
        "recall": {f"R@{k}": wavg(res["recall"][k]) for k in top_ks},
        "mrr": wavg(res["mrr"]),
        "by_category": {},
    }
    for cat_str, cat_res in res["by_category"].items():
        m_sum["by_category"][cat_str] = {
            "name": CATEGORY_NAMES.get(cat_str, f"Cat{cat_str}"),
            "n": sum(w for _, w in cat_res[top_ks[0]]),
            "recall": {f"R@{k}": wavg(cat_res[k]) for k in top_ks},
        }
    summary["methods"][method] = m_sum

with open("tests/eval_report_final.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n=== 全量结果（{total_qa} QA）===")
# 对比基线
baselines = {"fts": 0.108, "vector": 0.341, "hybrid": 0.391}
for method, mv in summary["methods"].items():
    r5 = mv["recall"]["R@5"]
    mrr = mv["mrr"]
    base = baselines.get(method, 0)
    delta = r5 - base
    print(f"[{method}] R@5={r5:.3f} MRR={mrr:.3f}  vs baseline {base:.3f} ({delta:+.3f})")
    for cat_str, cv in sorted(mv["by_category"].items()):
        print(f"  Cat{cat_str}: R@5={cv['recall']['R@5']:.3f} ({cv['n']} QA)")
print("\n报告已写入 tests/eval_report_final.json")
EOF
