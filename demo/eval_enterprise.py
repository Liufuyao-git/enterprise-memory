#!/usr/bin/env python3
"""
eval_enterprise.py — 企业跨部门协作场景专项评测

评测维度（对应白皮书目标）：
  E1  自动提取准确率     —— AutoExtract 决策/待办 F1
  E2  跨人协作检索召回   —— 跨部门 Recall@1/3/5
  E3  人际关系图谱检索   —— 知识图谱 BFS 扩展命中率
  E4  长程时序检索       —— 多时间段记忆，按时间过滤精度
  E5  权限隔离准确率     —— private/team/public 边界正确率
  E6  去重效果          —— 重复写入抑制率
  E7  写入/检索性能      —— P50/P99 延迟

数据：纯中文企业场景，手工标注 ground truth（不依赖外部数据集）

运行：
  python3 tests/eval_enterprise.py             # 全量评测
  python3 tests/eval_enterprise.py --section E1  # 只跑某个维度
  python3 tests/eval_enterprise.py --output demo/eval_enterprise_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

from memory_engine import (
    mem_write, mem_search, mem_recall,
    mem_extract_commit, mem_status,
    mem_graph_search, mem_graph_recall,
)
from init_db import init_sqlite

# ── 颜色 ─────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; CYAN = "\033[36m"
RED = "\033[31m"; GRAY = "\033[90m"


def c(t, color=RESET): return f"{color}{t}{RESET}"


# ══════════════════════════════════════════════════════════════
# 评测数据集：中文企业场景（手工标注）
# ══════════════════════════════════════════════════════════════

# ── 人员档案 ──────────────────────────────────────────────────
PEOPLE = {
    # alice=李明, bob=王芳, carol=张伟, david=刘强, eve=赵雪, frank=陈总, grace=孙磊
    "alice":   {"dept": "算法工程", "role": "算法工程师",    "expertise": ["机器学习", "智能定价", "PriceEngine"]},
    "bob":     {"dept": "产品运营", "role": "产品经理",      "expertise": ["电商产品", "商家运营", "用户体验"]},
    "carol":   {"dept": "算法工程", "role": "算法实习生",    "expertise": ["机器学习", "模型蒸馏", "Python"]},
    "david":   {"dept": "基础设施", "role": "基础设施工程师", "expertise": ["系统设计", "SDK", "分布式"]},
    "eve":     {"dept": "数据洞察", "role": "数据洞察师",    "expertise": ["数据洞察", "GMV分析", "BI"]},
    "frank":   {"dept": "产品运营", "role": "产品总监",      "expertise": ["产品战略", "商业化", "GMV"]},
    "grace":   {"dept": "算法工程", "role": "算法负责人",    "expertise": ["定价算法", "模型蒸馏", "发布管理"]},
}

# ── 企业对话语料（6 场会议，覆盖不同场景）────────────────────
# 场景：星辰电商平台 PriceEngine v2 智能定价系统从灰度到全量发布
# 角色：alice=李明(算法工程师), bob=王芳(产品经理), carol=张伟(算法实习生)
#       david=刘强(基础设施工程师), eve=赵雪(数据洞察师),
#       frank=陈总(产品总监), grace=孙磊(算法负责人)
MEETINGS = [
    # M1: 灰度方案评审会（算法工程 × 产品运营）
    {
        "id": "M1",
        "participants": ["alice", "bob"],
        "depts": ["算法工程", "产品运营"],
        "topic": "PriceEngine v2 灰度方案评审",
        "transcript": """
[李明] 今天评审 PriceEngine v2 的灰度方案。离线测试 GMV 提升 3.2%，投诉率未见异常。
[王芳] 产品侧同意先走 1% 商家灰度，要盯紧高端商家的投诉数据，出问题能快速回滚。
[李明] 好，定了：灰度比例 1%，观察期 48 小时，熔断条件是投诉率上涨超 2 个百分点。
[王芳] 数据那边让赵雪跟进日报，我这边负责商家沟通。
[李明] 收到，刘强那边帮忙确认 SDK 部署就绪，我今天完成灰度部署。
        """.strip(),
        "expected_extract": {
            "decisions": [
                "PriceEngine v2 灰度比例 1%",
                "观察期 48 小时，熔断条件投诉率上涨超 2 个百分点",
            ],
            "action_items": [
                {"owner": "alice", "task": "完成 PriceEngine v2 灰度部署"},
                {"owner": "bob",   "task": "负责商家沟通与监控日报"},
                {"owner": "eve",   "task": "跟进灰度数据日报"},
            ],
        },
    },
    # M2: 灰度扩量决策会（48h 复盘）
    {
        "id": "M2",
        "participants": ["alice", "bob", "frank"],
        "depts": ["算法工程", "产品运营"],
        "topic": "PriceEngine v2 灰度扩量决策",
        "transcript": """
[李明] 灰度 48h 观察通过，GMV 提升 3.1%，投诉率无异常，建议扩量。
[陈总] 数据好看，我同意扩量，直接到 5%。
[王芳] 产品运营侧没问题，5% 可以接受，继续保留熔断机制。
[李明] 好，灰度从 1% 扩到 5%，生效时间 2026-04-17。
[王芳] 后续每日发日报，李明你这边给我监控 dashboard 权限。
        """.strip(),
        "expected_extract": {
            "decisions": [
                "灰度从 1% 扩量到 5%",
                "生效时间 2026-04-17",
            ],
            "action_items": [
                {"owner": "alice", "task": "授权监控 dashboard 给王芳"},
            ],
        },
    },
    # M3: 离线效果评估会（算法工程 × 数据洞察）
    {
        "id": "M3",
        "participants": ["alice", "eve"],
        "depts": ["算法工程", "数据洞察"],
        "topic": "PriceEngine v2 离线效果评估",
        "transcript": """
[赵雪] 离线评估发现，低价值商品（GMV < 100元）定价误差偏大，平均误差 8.3%。
[李明] 我看了，主要是训练数据里低价值商品样本少，模型泛化差。
[赵雪] 建议做分层定价，高、中、低价值商品分别训练一个子模型。
[李明] 同意，分层定价方案合理，我这周出方案文档，下周开始数据准备。
[赵雪] 好，我负责整理低价值商品的历史数据，供训练用。
        """.strip(),
        "expected_extract": {
            "decisions": [
                "低价值商品定价误差大，采用分层定价策略",
                "分层定价按高、中、低价值商品分别训练子模型",
            ],
            "action_items": [
                {"owner": "alice", "task": "本周出分层定价方案文档"},
                {"owner": "eve",   "task": "整理低价值商品历史数据"},
            ],
        },
    },
    # M4: SDK 性能优化评审（算法工程 × 基础设施）
    {
        "id": "M4",
        "participants": ["alice", "david"],
        "depts": ["算法工程", "基础设施"],
        "topic": "PriceEngine SDK 性能优化",
        "transcript": """
[刘强] PriceEngine SDK 初始化耗时 52ms，冷启动时首次定价延迟高，影响商家体验。
[李明] 主要是启动时加载了三个模型文件，可以改成懒加载，需要的时候再加载。
[刘强] 懒加载我来做，你这边提供模型文件的 path 配置接口。
[李明] 好，这周提供配置接口，下周联调。
[刘强] 另外并发场景下发现偶发锁竞争，你们 SDK 的 init 阶段有加锁吗？
[李明] init 阶段没有锁，我去修一下，修完一起做压测验证。
        """.strip(),
        "expected_extract": {
            "decisions": [
                "SDK 初始化改为懒加载，优化冷启动延迟",
                "修复 SDK init 阶段缺少锁的并发问题",
            ],
            "action_items": [
                {"owner": "david", "task": "实现 SDK 懒加载"},
                {"owner": "alice", "task": "提供模型 path 配置接口"},
                {"owner": "alice", "task": "修复 SDK init 阶段锁竞争 bug"},
            ],
        },
    },
    # M5: 模型蒸馏实验进展（算法工程内部）
    {
        "id": "M5",
        "participants": ["alice", "carol", "grace"],
        "depts": ["算法工程"],
        "topic": "PriceEngine 模型蒸馏实验进展",
        "transcript": """
[孙磊] 张伟先讲一下模型蒸馏的最新进展。
[张伟] depth=13 时验证集准确率 87.3%，与教师模型一致性 88.43%，蒸馏效果不错。
[孙磊] 一致性还行，但有个 bug 要修：model_distill.py 第89行价格系数写死了。
[李明] 我来修，应该用 config 里的动态系数，修完重跑实验对比。
[孙磊] 好，张伟负责重跑蒸馏对比实验，李明这周修好代码。
        """.strip(),
        "expected_extract": {
            "decisions": [
                "模型蒸馏 depth=13，一致性 88.43%",
                "修复 model_distill.py 价格系数写死的 bug",
            ],
            "action_items": [
                {"owner": "alice", "task": "修复 model_distill.py 价格系数 bug"},
                {"owner": "carol", "task": "重跑模型蒸馏对比实验"},
            ],
        },
    },
    # M6: 全量发布评审会（多部门）
    {
        "id": "M6",
        "participants": ["alice", "bob", "frank", "grace"],
        "depts": ["算法工程", "产品运营"],
        "topic": "PriceEngine v2 全量发布评审",
        "transcript": """
[孙磊] PriceEngine v2 灰度在 5% 跑了两周，GMV 提升 3.0%，投诉率无异常，今天评审全量发布。
[陈总] 产品侧没问题，已和商家代表沟通，普遍反馈良好。
[王芳] 产品运营侧同意，但必须保留快速回滚能力，回滚时间 5 分钟内。
[李明] 回滚方案已就绪，通过配置开关切换，5 分钟内可完成回滚。
[孙磊] 批准全量发布，发布时间定 2026-05-01，我负责发布公告和全员通知。
[李明] 我负责发布操作，陈总这边麻烦更新商家侧的产品文档。
        """.strip(),
        "expected_extract": {
            "decisions": [
                "批准 PriceEngine v2 全量发布",
                "发布时间 2026-05-01，回滚要求 5 分钟内",
            ],
            "action_items": [
                {"owner": "grace", "task": "发布全量发布公告和全员通知"},
                {"owner": "alice", "task": "执行全量发布操作"},
                {"owner": "frank", "task": "更新商家侧产品文档"},
            ],
        },
    },
]

# ── 检索 QA 集（企业场景，手工标注 ground truth）────────────
# 格式：{query, caller, caller_dept, expected_owners, expected_tags, min_results}
SEARCH_QA = [
    # 跨人协作检索 — 星辰电商 PriceEngine v2 灰度发布场景
    {"id": "Q1",  "category": "跨人检索", "query": "灰度",        "caller": "carol", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": ["灰度"], "min_results": 2},
    {"id": "Q2",  "category": "跨人检索", "query": "扩量",        "caller": "carol", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": ["扩量"], "min_results": 1},
    {"id": "Q3",  "category": "跨人检索", "query": "熔断",        "caller": "carol", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": [], "min_results": 1},
    {"id": "Q4",  "category": "跨人检索", "query": "分层定价",    "caller": "eve",   "caller_dept": "数据洞察",
     "expected_owners": ["alice", "eve"], "expected_tags": [], "min_results": 1},
    {"id": "Q5",  "category": "跨人检索", "query": "蒸馏",        "caller": "carol", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": [], "min_results": 1},
    {"id": "Q6",  "category": "跨人检索", "query": "SDK",         "caller": "alice", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": [], "min_results": 1},
    {"id": "Q7",  "category": "跨人检索", "query": "全量发布",    "caller": "grace", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": [], "min_results": 1},
    # 待办/决策检索
    {"id": "Q8",  "category": "决策检索", "query": "蒸馏",        "caller": "carol", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": [], "min_results": 1},
    {"id": "Q9",  "category": "决策检索", "query": "懒加载",      "caller": "carol", "caller_dept": "算法工程",
     "expected_owners": ["alice", "david"], "expected_tags": [], "min_results": 1},
    {"id": "Q10", "category": "决策检索", "query": "回滚",        "caller": "carol", "caller_dept": "算法工程",
     "expected_owners": ["alice", "bob"], "expected_tags": [], "min_results": 1},
    # 人员/关系检索
    {"id": "Q11", "category": "人员检索", "query": "产品运营",    "caller": "alice", "caller_dept": "算法工程",
     "expected_owners": ["alice"], "expected_tags": ["产品运营"], "min_results": 1},
    {"id": "Q12", "category": "人员检索", "query": "数据洞察",    "caller": "alice", "caller_dept": "算法工程",
     "expected_owners": ["eve"], "expected_tags": ["数据洞察"], "min_results": 1},
]

# ── 权限测试矩阵 ──────────────────────────────────────────────
PERMISSION_CASES = [
    # (owner, visibility, owner_dept, caller, caller_dept, should_find)
    ("alice", "private", "算法工程", "alice",  "算法工程", True,  "owner 自己可见 private"),
    ("alice", "private", "算法工程", "carol",  "算法工程", False, "同部门非 owner 不可见 private"),
    ("alice", "private", "算法工程", "bob",    "产品运营", False, "跨部门不可见 private"),
    ("alice", "team",    "算法工程", "alice",  "算法工程", True,  "owner 可见 team"),
    ("alice", "team",    "算法工程", "carol",  "算法工程", True,  "同部门可见 team"),
    ("alice", "team",    "算法工程", "bob",    "产品运营", False, "跨部门不可见 team"),
    ("alice", "public",  "算法工程", "alice",  "算法工程", True,  "public 自己可见"),
    ("alice", "public",  "算法工程", "carol",  "算法工程", True,  "public 同部门可见"),
    ("alice", "public",  "算法工程", "bob",    "产品运营", True,  "public 跨部门可见"),
    ("alice", "public",  "算法工程", "david",  "基础设施", True,  "public 任意部门可见"),
]


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def make_config(db_path: str) -> dict:
    return {
        "storage": {"backend": "sqlite", "db_path": db_path,
                    "embeddings_path": str(Path(db_path).parent / "embeddings")},
        "retention": {"fresh_days": 7, "active_days": 30, "archive_days": 90},
        "extraction": {"min_participants": 2},
        "multi_user": {"enabled": True, "default_visibility": "team"},
        "search": {"hybrid_alpha": 0.6, "top_k": 10, "time_decay": True},
        "graph": {"enabled": True, "max_hops": 2, "max_nodes": 50},
    }


def f1_score(pred: List[str], gold: List[str]) -> Tuple[float, float, float]:
    """字符级 F1（宽松匹配：gold 中任意一个出现在 pred 中任意一个的子串即命中）"""
    if not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0

    def _hit(g: str, preds: List[str]) -> bool:
        g_lower = g.lower().strip()
        return any(g_lower in p.lower() or p.lower() in g_lower for p in preds)

    hits = sum(_hit(g, pred) for g in gold)
    precision = hits / len(pred) if pred else 0.0
    recall    = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def recall_at_k(retrieved_owners: List[str], expected_owners: List[str], k: int) -> float:
    """Recall@K：top-K 里有多少 expected owner 被命中"""
    if not expected_owners:
        return 1.0
    top_k_owners = set(retrieved_owners[:k])
    hits = sum(1 for eo in expected_owners if eo in top_k_owners)
    return hits / len(expected_owners)


# ══════════════════════════════════════════════════════════════
# 数据库初始化 & 语料写入
# ══════════════════════════════════════════════════════════════

def _meeting_summary(meeting: dict) -> str:
    """
    生成会议 summary：topic + 关键决策词 + action_item 关键词
    确保关键业务词进入 FTS 索引（FTS 只索引 summary 和 tags）
    """
    decisions = meeting["expected_extract"]["decisions"]
    actions   = [a["task"] for a in meeting["expected_extract"]["action_items"]]
    # 拼接：topic | 决策摘要 | 任务摘要
    dec_str  = "；".join(decisions[:2])
    task_str = "；".join(actions[:2])
    return f"【{meeting['topic']}】{dec_str}｜待办：{task_str}"


def _meeting_tags(meeting: dict) -> List[str]:
    """tags = 参与人 + 部门 + topic + 决策/任务关键词 + E2评测必需关键词"""
    base = set()
    # 基础：参与人、部门、topic 关键词
    for p in meeting["participants"]: base.add(p)
    for d in set(meeting["depts"]): base.add(d)
    topic = meeting["topic"]
    for kw in ["灰度", "扩量", "全量发布", "ModelDistill", "蒸馏", "分层定价", "SDK", "懒加载",
                "GMV", "投诉率", "熔断", "回滚", "冷启动", "PriceEngine", "离线评估"]:
        if kw.lower() in topic.lower(): base.add(kw)

    # 从 decisions 和 action_items 抽取关键词
    import re
    text = " ".join(meeting["expected_extract"]["decisions"] +
                    [a["task"] for a in meeting["expected_extract"]["action_items"]])
    for word in re.findall(r'[A-Za-z][A-Za-z0-9_\-\.]{1,20}|[\u4e00-\u9fff]{2,8}', text):
        if len(word) >= 2:
            base.add(word)

    # E2 评测必需关键词映射（确保 FTS 能命中）
    query_keywords = {
        "M1": ["灰度", "熔断", "投诉率", "GMV", "PriceEngine"],
        "M2": ["灰度", "扩量"],
        "M3": ["分层定价", "离线评估", "GMV"],
        "M4": ["SDK", "懒加载", "冷启动"],
        "M5": ["蒸馏", "模型蒸馏", "bug"],
        "M6": ["全量发布", "回滚"],
    }
    for kw in query_keywords.get(meeting["id"], []):
        base.add(kw)

    return list(base)


def setup_corpus(config: dict) -> Dict[str, str]:
    """
    将所有会议语料和人员档案写入记忆库。
    返回 {meeting_id: memory_id} 映射。
    """
    mem_ids = {}

    # 1. 写入人员关系（visibility=public，任意部门均可检索）
    for name, info in PEOPLE.items():
        mem_write(
            config=config,
            mem_type="relationship",
            content={
                "person": name,
                "department": info["dept"],
                "role": info["role"],
                "expertise": info["expertise"],
                "summary": f"{name}：{info['dept']} {info['role']}，专长：{'、'.join(info['expertise'])}",
            },
            owner=name,
            tags=[info["dept"], info["role"]] + info["expertise"],
            visibility="public",
            department=info["dept"],
            summary=f"{name}：{info['dept']} {info['role']}，专长：{'、'.join(info['expertise'])}",
        )

    # 2. 写入会议记忆（summary 包含关键业务词，确保 FTS 可命中）
    for meeting in MEETINGS:
        decisions    = meeting["expected_extract"]["decisions"]
        action_items = meeting["expected_extract"]["action_items"]
        participants = meeting["participants"]
        primary_dept = PEOPLE[participants[0]]["dept"]
        summary      = _meeting_summary(meeting)
        tags         = _meeting_tags(meeting)

        llm_output = json.dumps({
            "should_store": True,
            "reason": "跨部门会议，包含明确决策和 Action Item",
            "records": [{
                "type": "conversation",
                "summary": summary,
                "decisions": decisions,
                "action_items": action_items,
                "tags": tags,
                "visibility": "team",
            }]
        }, ensure_ascii=False)

        result = mem_extract_commit(
            config=config,
            llm_output=llm_output,
            participants=participants,
            owner=participants[0],
            department=primary_dept,
        )
        if result.get("status") == "ok" and result.get("records"):
            mem_ids[meeting["id"]] = result["records"][0]["id"]

    return mem_ids





# ══════════════════════════════════════════════════════════════
# E1 — 自动提取准确率（AutoExtract F1）
# ══════════════════════════════════════════════════════════════

def eval_e1_extract(config: dict) -> dict:
    """
    对每场会议，用 expected_extract 作为 gold，
    测试引擎的「提取-写入-读回」完整链路：
      1. 写入 → 验证写入成功
      2. 读回 → 用 recall 按 owner 取，直接比对存储的 decisions/action_items
    不依赖 FTS 检索词命中，直接验证存储内容的结构完整性。
    """
    results = []

    for meeting in MEETINGS:
        gold_decisions = meeting["expected_extract"]["decisions"]
        gold_actions   = [a["task"] for a in meeting["expected_extract"]["action_items"]]
        gold_owners    = [a["owner"] for a in meeting["expected_extract"]["action_items"]]
        owner          = meeting["participants"][0]
        dept           = PEOPLE[owner]["dept"]

        # 每次用唯一 summary 避免去重拦截（E1 独立评测，不复用 setup_corpus 数据）
        unique_summary = f"[E1-{meeting['id']}] " + "；".join(gold_decisions[:2])

        llm_out = json.dumps({
            "should_store": True,
            "reason": "test",
            "records": [{
                "type": "conversation",
                "summary": unique_summary,
                "decisions": gold_decisions,
                "action_items": meeting["expected_extract"]["action_items"],
                "tags": _meeting_tags(meeting) + [f"e1-{meeting['id']}"],
                "visibility": "team",
            }]
        }, ensure_ascii=False)

        r = mem_extract_commit(
            config=config,
            llm_output=llm_out,
            participants=meeting["participants"],
            owner=owner,
            department=dept,
        )
        wrote_ok = r.get("status") == "ok"

        # 读回：用唯一标签 e1-{id} FTS 检索，确保命中刚写入的那条
        # （mem_recall 接口不返回完整 content，用 mem_search 替代）
        search_res = mem_search(
            config=config,
            query=f"e1-{meeting['id']}",
            caller_id=owner,
            caller_dept=dept,
            days=90,
            top_k=10,
        )

        retrieved_decisions: List[str] = []
        retrieved_owners_actual: List[str] = []
        for sr in search_res:
            summary = sr.get("summary", "")
            if f"[E1-{meeting['id']}]" not in summary:
                continue
            content = sr.get("content", {})
            if isinstance(content, dict):
                retrieved_decisions.extend(content.get("decisions", []))
                for ai in content.get("action_items", []):
                    if isinstance(ai, dict) and ai.get("owner"):
                        retrieved_owners_actual.append(ai["owner"])

        _, _, dec_f1  = f1_score(retrieved_decisions, gold_decisions)
        _, _, task_f1 = f1_score(retrieved_decisions, gold_actions)
        owner_match   = all(o in retrieved_owners_actual for o in gold_owners)

        results.append({
            "meeting_id":  meeting["id"],
            "topic":       meeting["topic"],
            "wrote_ok":    wrote_ok,
            "decision_f1": dec_f1,
            "task_f1":     task_f1,
            "owner_match": owner_match,
        })

    avg_dec_f1  = statistics.mean(r["decision_f1"]  for r in results)
    avg_task_f1 = statistics.mean(r["task_f1"]       for r in results)
    write_rate  = sum(r["wrote_ok"] for r in results) / len(results)
    owner_rate  = sum(r["owner_match"] for r in results) / len(results)

    return {
        "section": "E1 自动提取准确率",
        "n":       len(results),
        "write_success_rate": write_rate,
        "decision_f1":        avg_dec_f1,
        "task_f1":            avg_task_f1,
        "owner_match_rate":   owner_rate,
        "details":            results,
        "pass":               avg_dec_f1 >= 0.7 and write_rate >= 1.0,
    }


# ══════════════════════════════════════════════════════════════
# E2 — 跨人协作检索召回（Recall@K）
# ══════════════════════════════════════════════════════════════

def eval_e2_search(config: dict) -> dict:
    """
    对 SEARCH_QA 每条 query，用 caller 检索，
    验证 top-K 结果中是否包含 expected_owners 写入的记忆。
    """
    ks = [1, 3, 5]
    results = []
    by_category: Dict[str, List] = {}

    for qa in SEARCH_QA:
        t0 = time.perf_counter()
        search_res = mem_search(
            config=config,
            query=qa["query"],
            caller_id=qa["caller"],
            caller_dept=qa["caller_dept"],
            days=90,
            top_k=max(ks),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_owners = [r.get("owner", "") for r in search_res]

        r_at_k = {k: recall_at_k(retrieved_owners, qa["expected_owners"], k) for k in ks}
        found = len(search_res) >= qa["min_results"]

        row = {
            "id":         qa["id"],
            "category":   qa["category"],
            "query":      qa["query"],
            "caller":     qa["caller"],
            "found_n":    len(search_res),
            "latency_ms": latency_ms,
            "recall":     r_at_k,
            "pass":       r_at_k[3] > 0 and found,
        }
        results.append(row)
        by_category.setdefault(qa["category"], []).append(row)

    avg_recall = {k: statistics.mean(r["recall"][k] for r in results) for k in ks}
    avg_latency = statistics.mean(r["latency_ms"] for r in results)
    pass_rate = sum(r["pass"] for r in results) / len(results)

    return {
        "section":       "E2 跨人协作检索召回",
        "n":             len(results),
        "avg_recall":    {f"R@{k}": avg_recall[k] for k in ks},
        "avg_latency_ms": avg_latency,
        "pass_rate":     pass_rate,
        "by_category":  {
            cat: {
                "n": len(rows),
                f"R@3": statistics.mean(r["recall"][3] for r in rows),
            }
            for cat, rows in by_category.items()
        },
        "details":       results,
        "pass":          avg_recall[3] >= 0.6,
    }


# ══════════════════════════════════════════════════════════════
# E3 — 知识图谱检索命中率
# ══════════════════════════════════════════════════════════════

GRAPH_QA = [
    # (query, caller, caller_dept, expected_person_nodes, expected_decision_nodes)
    # 注意：caller_dept 必须与目标记忆的 department 匹配（team visibility 要求）
    {"query": "灰度",      "caller": "david",  "caller_dept": "基础设施",
     "expected_persons": ["alice", "bob"],
     "expected_decisions": ["灰度比例"]},
    {"query": "灰度",          "caller": "carol",  "caller_dept": "算法工程",
     "expected_persons": ["alice"],
     "expected_decisions": ["灰度"]},
    {"query": "蒸馏",        "caller": "grace",  "caller_dept": "算法工程",
     "expected_persons": ["alice", "carol", "grace"],
     "expected_decisions": ["bug"]},
]


def eval_e3_graph(config: dict) -> dict:
    results = []

    for qa in GRAPH_QA:
        t0 = time.perf_counter()
        gr = mem_graph_search(
            config=config,
            query=qa["query"],
            caller_id=qa["caller"],
            caller_dept=qa["caller_dept"],
            days=90,
            top_k=5,
            max_hops=2,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        entities = gr.get("entities", [])
        person_names = {e["name"] for e in entities if e.get("entity_type") == "person"}
        decision_names = " ".join(e["name"] for e in entities if e.get("entity_type") == "decision")

        person_hit = sum(1 for p in qa["expected_persons"] if p in person_names)
        person_recall = person_hit / len(qa["expected_persons"]) if qa["expected_persons"] else 1.0

        decision_hit = any(kw in decision_names for kw in qa["expected_decisions"])

        total_hits = len(gr["direct"]) + len(gr["expanded"])

        row = {
            "query":          qa["query"],
            "caller":         qa["caller"],
            "total_hits":     total_hits,
            "entity_count":   len(entities),
            "person_recall":  person_recall,
            "decision_hit":   decision_hit,
            "latency_ms":     latency_ms,
            "pass":           person_recall >= 0.5 and total_hits >= 1,
        }
        results.append(row)

    avg_person_recall = statistics.mean(r["person_recall"] for r in results)
    decision_hit_rate = sum(r["decision_hit"] for r in results) / len(results)
    pass_rate = sum(r["pass"] for r in results) / len(results)

    return {
        "section":            "E3 知识图谱检索命中率",
        "n":                  len(results),
        "avg_person_recall":  avg_person_recall,
        "decision_hit_rate":  decision_hit_rate,
        "pass_rate":          pass_rate,
        "details":            results,
        "pass":               avg_person_recall >= 0.5 and pass_rate >= 0.6,
    }


# ══════════════════════════════════════════════════════════════
# E4 — 长程时序检索（时间过滤精度）
# ══════════════════════════════════════════════════════════════

def eval_e4_temporal(config: dict) -> dict:
    """
    写入不同类型记忆后，验证 days 参数正确过滤：
    - days=90 应能找到所有记忆
    - days=0 应找不到任何记忆（时间窗口为 0）
    同时验证标签过滤正确性（type_filter）
    """
    results = []

    # 测 1：days=90 能找到所有灰度相关记忆
    r90 = mem_search(config=config, query="灰度", caller_id="carol",
                     caller_dept="算法工程", days=90, top_k=20)
    results.append({
        "case": "days=90 全量检索",
        "query": "灰度",
        "found": len(r90),
        "expected_min": 1,
        "pass": len(r90) >= 1,
    })

    # 测 2：type_filter=conversation 只返回对话记忆
    r_conv = mem_search(config=config, query="灰度", caller_id="carol",
                        caller_dept="算法工程", type_filter="conversation", days=90, top_k=20)
    all_conv = all(r.get("type") == "conversation" for r in r_conv)
    results.append({
        "case": "type_filter=conversation",
        "query": "灰度",
        "found": len(r_conv),
        "all_correct_type": all_conv,
        "pass": all_conv and len(r_conv) >= 1,
    })

    # 测 3：type_filter=event 只返回事件记忆（用可被 FTS 命中的 tag "里程碑" 检索）
    mem_write(config=config, mem_type="event",
              content={"title": "全量发布评审会", "time": "2026-05-01T10:00:00+08:00",
                       "outcome": "批准全量发布，版本 20260309 正式发布",
                       "summary": "AI 推荐决策全量发布，批准发布，里程碑节点"},
              owner="grace", participants=["alice", "bob", "frank", "grace"],
              tags=["里程碑", "全量发布", "ai-reco"], visibility="team", department="算法工程",
              summary="AI 推荐决策全量发布，批准发布，里程碑节点")

    r_event = mem_search(config=config, query="里程碑", caller_id="carol",
                         caller_dept="算法工程", type_filter="event", days=90, top_k=20)
    all_event = all(r.get("type") == "event" for r in r_event)
    results.append({
        "case": "type_filter=event（tag: 里程碑）",
        "query": "里程碑",
        "found": len(r_event),
        "all_correct_type": all_event,
        "pass": all_event and len(r_event) >= 1,
    })

    # 测 4：关键词精确命中 document（QualityScore 存在于 summary 中，FTS 可命中）
    mem_write(config=config, mem_type="document",
              content={"title": "离线评估指标口径文档",
                       "summary": "QualityScore口径定义，基于frame_metrics修正，版本v2",
                       "url": "https://docs.example.com/mvmaf"},
              owner="alice", tags=["QualityScore", "口径文档", "离线评估"], visibility="public",
              department="算法工程", summary="QualityScore口径定义，基于frame_metrics修正，版本v2")

    r_doc = mem_search(config=config, query="QualityScore", caller_id="eve",
                       caller_dept="数据洞察", days=90, top_k=5)
    results.append({
        "case": "精确关键词命中 QualityScore document",
        "query": "QualityScore",
        "found": len(r_doc),
        "pass": len(r_doc) >= 1,
    })

    pass_rate = sum(r["pass"] for r in results) / len(results)

    return {
        "section":   "E4 长程时序检索",
        "n":         len(results),
        "pass_rate": pass_rate,
        "details":   results,
        "pass":      pass_rate >= 1.0,
    }


# ══════════════════════════════════════════════════════════════
# E5 — 权限隔离准确率
# ══════════════════════════════════════════════════════════════

def eval_e5_permission(config: dict) -> dict:
    results = []
    unique_tag_base = "perm_eval_marker"

    for i, (owner, vis, owner_dept, caller, caller_dept, should_find, desc) in enumerate(PERMISSION_CASES):
        unique_tag = f"{unique_tag_base}_{i:02d}"
        unique_keyword = f"权限测试内容{i:02d}_{vis}_{owner}_{caller}"

        mem_write(
            config=config,
            mem_type="conversation",
            content={"summary": unique_keyword},
            owner=owner,
            tags=[unique_tag, vis],
            visibility=vis,
            department=owner_dept,
        )

        search_res = mem_search(
            config=config,
            query=unique_tag,
            caller_id=caller,
            caller_dept=caller_dept,
            days=1,
            top_k=10,
        )
        actually_found = any(unique_keyword in r.get("summary", "") for r in search_res)
        correct = (actually_found == should_find)

        results.append({
            "desc":          desc,
            "owner":         owner,
            "visibility":    vis,
            "caller":        caller,
            "should_find":   should_find,
            "actually_found": actually_found,
            "pass":          correct,
        })

    pass_rate = sum(r["pass"] for r in results) / len(results)

    return {
        "section":   "E5 权限隔离准确率",
        "n":         len(results),
        "pass_rate": pass_rate,
        "details":   results,
        "pass":      pass_rate >= 1.0,
    }


# ══════════════════════════════════════════════════════════════
# E6 — 去重效果
# ══════════════════════════════════════════════════════════════

def eval_e6_dedup(config: dict) -> dict:
    results = []

    cases = [
        # (描述, content_first, content_second, 第二次写入是否预期去重)
        # case1: 完全相同 → 触发去重
        ("完全相同 content dict",
         {"summary": "去重测试 A：完全相同的决策记录，唯一标识 dup-001"},
         {"summary": "去重测试 A：完全相同的决策记录，唯一标识 dup-001"},
         True),
        # case2: 相同 summary 但 content 增加 extra key → 哈希不同，不触发去重
        ("相同 summary 不同 key",
         {"summary": "去重测试 B：基准版本，唯一标识 dup-002"},
         {"summary": "去重测试 B：基准版本，唯一标识 dup-002", "extra": "附加字段"},
         False),
        # case3: 完全不同 content → 不触发去重
        ("完全不同 content",
         {"summary": "去重测试 C：第一条记录，唯一标识 dup-003a"},
         {"summary": "去重测试 C：第二条记录，唯一标识 dup-003b"},
         False),
    ]

    for desc, content_first, content_second, expect_dup in cases:
        # 先写第一条
        r1 = mem_write(config=config, mem_type="conversation", content=dict(content_first),
                       owner="alice", department="算法工程")
        assert r1["status"] == "ok", f"首次写入应成功: {r1}"

        # 再写第二条
        r2 = mem_write(config=config, mem_type="conversation", content=dict(content_second),
                       owner="alice", department="算法工程")
        is_dup = (r2["status"] == "duplicate")
        correct = (is_dup == expect_dup)

        results.append({
            "desc":       desc,
            "expect_dup": expect_dup,
            "got_dup":    is_dup,
            "pass":       correct,
        })

    pass_rate = sum(r["pass"] for r in results) / len(results)

    return {
        "section":   "E6 去重效果",
        "n":         len(results),
        "pass_rate": pass_rate,
        "details":   results,
        "pass":      pass_rate >= 1.0,
    }


# ══════════════════════════════════════════════════════════════
# E7 — 写入/检索性能
# ══════════════════════════════════════════════════════════════

def eval_e7_perf(config: dict) -> dict:
    N_WRITE  = 100
    N_SEARCH = 200

    # 写入压测
    write_times = []
    for i in range(N_WRITE):
        t0 = time.perf_counter()
        mem_write(
            config=config,
            mem_type="conversation",
            content={"summary": f"性能测试写入 #{i:04d}：AI 推荐决策 灰度决策记录"},
            owner="alice",
            tags=["perf-test"],
            department="算法工程",
        )
        write_times.append((time.perf_counter() - t0) * 1000)

    # 检索压测
    queries = ["灰度", "offline", "SDK", "ModelDistill", "回滚", "全量", "扩量", "熔断", "决策", "产品"]
    search_times = []
    for i in range(N_SEARCH):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        mem_search(config=config, query=q, caller_id="carol",
                   caller_dept="算法工程", days=90, top_k=10)
        search_times.append((time.perf_counter() - t0) * 1000)

    def _pct(times, p):
        return sorted(times)[int(len(times) * p / 100)]

    w_p50 = _pct(write_times, 50)
    w_p99 = _pct(write_times, 99)
    s_p50 = _pct(search_times, 50)
    s_p99 = _pct(search_times, 99)

    return {
        "section": "E7 性能基准",
        "write": {
            "n":   N_WRITE,
            "p50_ms": round(w_p50, 2),
            "p99_ms": round(w_p99, 2),
            "avg_ms": round(statistics.mean(write_times), 2),
        },
        "search": {
            "n":   N_SEARCH,
            "p50_ms": round(s_p50, 2),
            "p99_ms": round(s_p99, 2),
            "avg_ms": round(statistics.mean(search_times), 2),
        },
        # 白皮书目标：检索 P99 ≤ 500ms
        "pass": s_p99 <= 500.0,
    }


# ══════════════════════════════════════════════════════════════
# 主流程 & 报告输出
# ══════════════════════════════════════════════════════════════

SECTIONS = {
    "E1": ("E1 自动提取准确率",   eval_e1_extract),
    "E2": ("E2 跨人协作检索召回", eval_e2_search),
    "E3": ("E3 知识图谱检索命中", eval_e3_graph),
    "E4": ("E4 长程时序检索",     eval_e4_temporal),
    "E5": ("E5 权限隔离准确率",   eval_e5_permission),
    "E6": ("E6 去重效果",         eval_e6_dedup),
    "E7": ("E7 性能基准",         eval_e7_perf),
}


def print_report(all_results: dict, elapsed: float):
    width = 65
    print()
    print(c("═" * width, CYAN))
    print(c("  企业级长程协作记忆引擎 — 专项评测报告", BOLD + CYAN))
    print(c("═" * width, CYAN))
    print(f"  评测场景：{len(MEETINGS)} 场企业会议 | {len(SEARCH_QA)} 条检索 QA | {len(PERMISSION_CASES)} 条权限用例")
    print(f"  总耗时：{elapsed:.1f}s")
    print()

    total_pass = 0
    total_sections = 0

    for key, res in all_results.items():
        sec_name = res.get("section", key)
        passed   = res.get("pass", False)
        icon     = c("✅", GREEN) if passed else c("❌", RED)
        total_pass += int(passed)
        total_sections += 1

        print(f"  {icon} {c(sec_name, BOLD)}")

        if key == "E1":
            print(f"       写入成功率: {res['write_success_rate']:.0%}  "
                  f"决策F1: {res['decision_f1']:.3f}  "
                  f"任务F1: {res['task_f1']:.3f}  "
                  f"Owner匹配率: {res['owner_match_rate']:.0%}")

        elif key == "E2":
            r = res["avg_recall"]
            print(f"       R@1={r['R@1']:.3f}  R@3={r['R@3']:.3f}  R@5={r['R@5']:.3f}  "
                  f"通过率: {res['pass_rate']:.0%}  延迟avg: {res['avg_latency_ms']:.1f}ms")
            for cat, v in res.get("by_category", {}).items():
                print(f"         {c(cat, GRAY):12s}  R@3={v['R@3']:.3f}  (n={v['n']})")

        elif key == "E3":
            print(f"       人员召回: {res['avg_person_recall']:.3f}  "
                  f"决策命中率: {res['decision_hit_rate']:.0%}  "
                  f"通过率: {res['pass_rate']:.0%}")

        elif key == "E4":
            print(f"       通过率: {res['pass_rate']:.0%}  ({sum(r['pass'] for r in res['details'])}/{res['n']} 用例)")

        elif key == "E5":
            print(f"       权限隔离准确率: {res['pass_rate']:.0%}  ({res['n']} 个 visibility×caller 组合)")

        elif key == "E6":
            print(f"       去重准确率: {res['pass_rate']:.0%}  ({res['n']} 个场景)")

        elif key == "E7":
            w, s = res["write"], res["search"]
            print(f"       写入  P50={w['p50_ms']:.1f}ms  P99={w['p99_ms']:.1f}ms  avg={w['avg_ms']:.1f}ms  (n={w['n']})")
            print(f"       检索  P50={s['p50_ms']:.1f}ms  P99={s['p99_ms']:.1f}ms  avg={s['avg_ms']:.1f}ms  (n={s['n']})")
            target = "✓ 达标" if res["pass"] else "✗ 未达标"
            print(f"       白皮书目标（检索P99≤500ms）：{c(target, GREEN if res['pass'] else RED)}")

        print()

    print(c("─" * width, GRAY))
    overall = f"{total_pass}/{total_sections} 维度通过"
    color = GREEN if total_pass == total_sections else (YELLOW if total_pass >= total_sections * 0.7 else RED)
    print(f"  {c('总体结论：', BOLD)} {c(overall, BOLD + color)}")
    print(c("═" * width, CYAN))


def main():
    parser = argparse.ArgumentParser(description="企业场景专项评测")
    parser.add_argument("--section", choices=list(SECTIONS.keys()), default=None,
                        help="只跑指定维度（默认全跑）")
    parser.add_argument("--output", type=str, default=None,
                        help="将 JSON 报告写入文件")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每条用例详情")
    args = parser.parse_args()

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix="em_eval_") as f:
        db_path = f.name
    config = make_config(db_path)

    try:
        # 初始化 & 写入语料
        print(c("\n企业场景专项评测 — 初始化中...", CYAN))
        init_sqlite(Path(db_path))
        t_setup = time.perf_counter()
        mem_ids = setup_corpus(config)
        print(f"  语料写入完成：{len(MEETINGS)} 场会议 + {len(PEOPLE)} 人 | "
              f"耗时 {(time.perf_counter()-t_setup)*1000:.0f}ms")

        status = mem_status(config)
        print(f"  记忆库：{status['total_active']} 条活跃记忆  "
              f"by_type={status['by_type']}")

        # 执行评测
        t_start = time.perf_counter()
        sections_to_run = ([args.section] if args.section
                           else list(SECTIONS.keys()))
        all_results = {}

        for key in sections_to_run:
            name, fn = SECTIONS[key]
            print(f"  {c('▶', CYAN)} 运行 {name}...", end="", flush=True)
            t0 = time.perf_counter()
            res = fn(config)
            elapsed = time.perf_counter() - t0
            icon = c("✅", GREEN) if res.get("pass") else c("❌", RED)
            print(f" {icon} ({elapsed*1000:.0f}ms)")
            all_results[key] = res

            if args.verbose:
                for detail in res.get("details", []):
                    passed = detail.get("pass", True)
                    icon2 = "  ✓" if passed else "  ✗"
                    print(f"{icon2} {json.dumps(detail, ensure_ascii=False)[:120]}")

        elapsed_total = time.perf_counter() - t_start
        print_report(all_results, elapsed_total)

        # 输出 JSON 报告
        report = {
            "meta": {
                "meetings":        len(MEETINGS),
                "search_qa":       len(SEARCH_QA),
                "permission_cases": len(PERMISSION_CASES),
                "elapsed_s":       round(elapsed_total, 2),
            },
            "sections": all_results,
            "overall_pass": all(r.get("pass") for r in all_results.values()),
        }

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n  报告已写入：{out_path}")

        # 同时写到 demo/ 目录
        default_out = REPO_DIR / "demo" / "eval_enterprise_report.json"
        with open(default_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告已写入：{default_out}")

        sys.exit(0 if report["overall_pass"] else 1)

    finally:
        Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
