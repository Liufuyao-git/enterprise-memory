#!/usr/bin/env python3
"""
demo_enterprise.py — 企业级长程协作记忆引擎端到端 Demo

演示场景：
  星辰电商 算法工程组 × 产品运营组 跨部门协作全链路

场景背景：
  星辰电商上线 AI 智能定价系统（PriceEngine v2），用机器学习替代规则定价，
  提升 GMV 同时控制利润率。

角色映射（caller_id 不变，只改展示名）：
  alice  → 李明（算法工程师）
  bob    → 王芳（产品经理）
  carol  → 张伟（算法实习生）
  david  → 刘强（基础设施工程师）
  eve    → 赵雪（数据洞察师）
  frank  → 陈总（产品总监）
  grace  → 孙磊（算法负责人）

场景一：PriceEngine v2 灰度方案评审会
  李明（算法工程）+ 王芳（产品运营）开会，讨论智能定价系统灰度发布方案
  → 自动提取会议决策 & 待办事项写入记忆库

场景二：跨部门决策查询
  张伟（算法实习生，新人）想了解"上次和产品侧的灰度决策是什么"
  → 语义检索，跨人找到 李明/王芳 的决策记录

场景三：知识图谱扩展检索
  刘强（基础设施组）想知道"PriceEngine 项目涉及哪些人"
  → 图谱 BFS 扩展，从项目实体找到关联人员和决策

场景四：人际关系召回
  李明 问："王芳 负责什么，我们合作过哪些事"
  → Recall 接口，找回 李明-王芳 协作链路

场景五：长程记忆验证（跨"会话"持久化）
  模拟新会话重启，张伟 在"新会话"里依然能找到灰度方案决策
  → 证明跨会话持久化能力

运行：
  python3 demo_enterprise.py              # 完整 Demo（使用临时 DB）
  python3 demo_enterprise.py --keep-db   # 保留数据库，可后续查询
  python3 demo_enterprise.py --verbose   # 打印详细检索结果
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# ── 路径设置 ─────────────────────────────────────────────────
# 支持从项目根目录或 demo/ 子目录运行
_SCRIPT_FILE = Path(__file__).resolve()
_REPO_DIR = _SCRIPT_FILE.parent if _SCRIPT_FILE.parent.name != "demo" else _SCRIPT_FILE.parent.parent
sys.path.insert(0, str(_REPO_DIR / "scripts"))

from memory_engine import (
    mem_write, mem_search, mem_recall,
    mem_extract_commit, mem_status,
    mem_graph_search, mem_graph_recall,
    load_config,
)
from init_db import init_sqlite

# ── 颜色输出 ─────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
GRAY   = "\033[90m"
BLUE   = "\033[34m"
MAGENTA = "\033[35m"


def c(text, color=RESET):
    return f"{color}{text}{RESET}"


def banner(title: str, color=CYAN):
    width = 64
    print()
    print(c("═" * width, color))
    print(c(f"  {title}", BOLD + color))
    print(c("═" * width, color))


def section(title: str):
    print(f"\n{c('▶', CYAN)} {c(title, BOLD)}")


def ok(msg: str):
    print(f"  {c('✅', GREEN)} {msg}")


def info(msg: str):
    print(f"  {c('ℹ', CYAN)}  {msg}")


def warn(msg: str):
    print(f"  {c('⚠', YELLOW)}  {msg}")


def result_item(idx: int, r: dict, verbose: bool = False):
    summary = r.get("summary", "（无摘要）")[:70]
    tags = r.get("tags", [])
    rtype = r.get("type", "?")
    owner = r.get("owner", "?")
    created = (r.get("created_at") or "")[:16]
    print(f"    {c(f'[{idx}]', GRAY)} {c(rtype, YELLOW)} | {summary}")
    print(f"         {c('owner:', GRAY)} {owner}  "
          f"{c('tags:', GRAY)} {tags}  "
          f"{c('time:', GRAY)} {created}")
    if verbose and r.get("content"):
        content = r["content"]
        if isinstance(content, dict):
            decisions = content.get("decisions", [])
            action_items = content.get("action_items", [])
            if decisions:
                for d in decisions:
                    print(f"         {c('决策:', BLUE)} {d}")
            if action_items:
                for a in action_items:
                    if isinstance(a, dict):
                        print(f"         {c('待办:', MAGENTA)} [{a.get('owner','?')}] "
                              f"{a.get('task','?')} (due: {a.get('due','?')})")


# ══════════════════════════════════════════════════════════════
# 测试数据：模拟跨部门会议对话文本 + LLM 提取输出
# 场景：星辰电商 PriceEngine v2 智能定价系统灰度上线
# ══════════════════════════════════════════════════════════════

# 场景一：M1 PriceEngine v2 灰度方案评审会
# 参与人：alice=李明（算法工程）× bob=王芳（产品运营）
MEETING_TRANSCRIPT_1 = """
[李明] 今天评审 PriceEngine v2 的灰度方案。模型离线测试 GMV 提升 3.2%，投诉率未见异常。
[王芳] 产品侧同意先走 1% 商家灰度，但要盯紧高端商家的投诉数据，出问题要能快速回滚。
[李明] 好，定了：灰度比例 1%，观察期 48 小时，熔断条件是投诉率上涨超 2 个百分点。
[王芳] 数据那边让赵雪跟进日报，我这边负责商家沟通。
[李明] 收到，刘强那边帮忙确认 SDK 部署就绪。
""".strip()

# 模拟 LLM 对 M1 的提取输出（实际场景由 OpenClaw 大模型自动生成）
EXTRACT_OUTPUT_1 = {
    "should_store": True,
    "reason": "包含明确灰度决策、熔断阈值约定和两个明确 Action Item，跨部门协作信号强",
    "records": [
        {
            "type": "conversation",
            "summary": "PriceEngine v2 灰度方案评审：灰度比例 1%，观察期 48h，熔断条件投诉率+2pp，GMV 提升 3.2%",
            "decisions": [
                "PriceEngine v2 灰度比例定为 1%",
                "观察期 48 小时，熔断条件：投诉率上涨超 2 个百分点即回滚",
                "48 小时后举行复盘会议，李明 + 王芳 参与",
            ],
            "action_items": [
                {"owner": "alice", "task": "确认 PriceEngine v2 灰度部署就绪", "due": "2026-04-15"},
                {"owner": "bob",   "task": "负责商家沟通，跟进高端商家投诉数据",   "due": "2026-04-16"},
                {"owner": "eve",   "task": "跟进投诉率与 GMV 日报",              "due": "2026-04-20"},
            ],
            "tags": ["灰度", "PriceEngine", "智能定价", "GMV", "熔断", "投诉率", "回滚", "跨部门"],
            "visibility": "team",
        }
    ],
}

# M2：灰度扩量决策会（48h 复盘，alice=李明 主导）
EXTRACT_OUTPUT_2 = {
    "should_store": True,
    "reason": "灰度观察期结束，包含扩量决策和新 Action Item",
    "records": [
        {
            "type": "conversation",
            "summary": "PriceEngine v2 灰度扩量：48h 无异常，投诉率稳定，扩量至 5%，生效时间 2026-04-17",
            "decisions": [
                "灰度 48h 观察通过，投诉率未见上涨，GMV 提升符合预期",
                "灰度比例从 1% 扩量到 5%，生效时间 2026-04-17",
            ],
            "action_items": [
                {"owner": "alice", "task": "更新灰度配置，从 1% 扩至 5%", "due": "2026-04-17"},
                {"owner": "eve",   "task": "持续跟进扩量后 GMV 和投诉率日报", "due": "2026-04-20"},
            ],
            "tags": ["灰度", "扩量", "PriceEngine", "GMV", "投诉率"],
            "visibility": "team",
        }
    ],
}

# 人际关系数据（caller_id 保持英文，展示名用中文）
RELATIONSHIP_DATA = [
    {
        "person": "alice",  # 李明
        "email": "liming@starcart.com",
        "department": "算法工程",
        "role": "算法工程师",
        "expertise": ["机器学习", "智能定价", "PriceEngine"],
        "summary": "alice（李明）：算法工程师，负责 PriceEngine v2 模型训练和推理",
    },
    {
        "person": "bob",    # 王芳
        "email": "wangfang@starcart.com",
        "department": "产品运营",
        "role": "产品经理",
        "expertise": ["电商产品", "商家运营", "GMV 管控"],
        "summary": "bob（王芳）：产品运营 PM，负责智能定价产品设计和商家沟通",
    },
    {
        "person": "carol",  # 张伟
        "email": "zhangwei@starcart.com",
        "department": "算法工程",
        "role": "算法实习生",
        "expertise": ["机器学习", "Python", "模型蒸馏"],
        "summary": "carol（张伟）：算法工程实习生，参与 PriceEngine 模型蒸馏实验",
    },
    {
        "person": "david",  # 刘强
        "email": "liuqiang@starcart.com",
        "department": "基础设施",
        "role": "基础设施工程师",
        "expertise": ["SDK 开发", "服务治理", "懒加载优化"],
        "summary": "david（刘强）：基础设施工程师，负责 PriceEngine SDK 部署和性能优化",
    },
]

# 里程碑事件
EVENT_DATA = [
    {
        "title": "PriceEngine v2 灰度启动",
        "time": "2026-04-15T10:00:00+08:00",
        "outcome": "1% 商家灰度正式上线，观察期开始",
        "summary": "PriceEngine v2 智能定价系统灰度正式启动，比例 1% 商家，GMV+3.2%",
        "participants": ["alice", "bob"],
    },
    {
        "title": "PriceEngine v2 灰度扩量：1% → 5%",
        "time": "2026-04-17T10:00:00+08:00",
        "outcome": "扩量至 5% 商家，观察期继续",
        "summary": "PriceEngine v2 灰度扩量至 5% 商家，48h 观察期无异常，决策通过",
        "participants": ["alice", "bob"],
    },
]

# 文档记忆
DOCUMENT_DATA = [
    {
        "title": "PriceEngine v2 离线效果评估报告",
        "url": "https://docs.starcart.com/doc/pricingengine_eval_v2",
        "summary": "PriceEngine v2 评估：GMV+3.2%，分层定价在低价值商品上误差偏大，需优化",
        "key_facts": [
            "整体 GMV 提升 3.2%",
            "高价值商品定价准确率 94%",
            "低价值商品预测误差偏大，建议分层定价",
            "投诉率较规则定价下降 1.1%",
        ],
        "tags": ["评估报告", "PriceEngine", "GMV", "分层定价"],
    },
    {
        "title": "PriceEngine v2 灰度方案文档",
        "url": "https://docs.starcart.com/doc/pricingengine_gray_plan",
        "summary": "PriceEngine v2 灰度方案：1%→5%→全量，熔断机制设计，回滚 5 分钟内完成",
        "key_facts": [
            "灰度路径：1% → 5% → 全量",
            "熔断条件：投诉率上涨超 2pp 或 GMV 下降超 5%",
            "回滚方式：配置开关切换，5 分钟内完成",
        ],
        "tags": ["灰度方案", "PriceEngine", "熔断", "回滚", "全量发布"],
    },
]


# ══════════════════════════════════════════════════════════════
# Demo 核心逻辑
# ══════════════════════════════════════════════════════════════

def make_config(db_path: str) -> dict:
    return {
        "storage": {
            "backend": "sqlite",
            "db_path": db_path,
            "embeddings_path": str(Path(db_path).parent / "embeddings"),
        },
        "retention": {"fresh_days": 7, "active_days": 30, "archive_days": 90},
        "extraction": {"min_participants": 2},
        "multi_user": {"enabled": True, "default_visibility": "team"},
        "search": {"hybrid_alpha": 0.6, "top_k": 10, "time_decay": True},
        "graph": {"enabled": True, "max_hops": 2, "max_nodes": 50},
    }


def run_demo(db_path: str, verbose: bool = False):
    config = make_config(db_path)
    init_sqlite(Path(db_path))

    # ══════════════════════════════════════════════════════════
    # 场景一：PriceEngine v2 灰度方案评审会 → 自动提取写入
    # 参与人：alice=李明（算法工程）× bob=王芳（产品运营）
    # ══════════════════════════════════════════════════════════
    banner("场景一：PriceEngine v2 灰度方案评审会 × 自动记忆提取")

    section("会议对话（alice=李明 × bob=王芳，算法工程 × 产品运营）")
    for line in MEETING_TRANSCRIPT_1.strip().split("\n"):
        if line.strip():
            print(f"  {c(line.strip(), GRAY)}")

    section("OpenClaw 自动触发 AutoExtract（会话结束后 LLM 提取）")
    info("检测到跨部门对话（2 人参与，含决策信号）→ 触发自动提取")
    time.sleep(0.2)

    # 提交 LLM 提取结果（模拟 OpenClaw 大模型调用返回）
    r1 = mem_extract_commit(
        config=config,
        llm_output=json.dumps(EXTRACT_OUTPUT_1, ensure_ascii=False),
        participants=["alice", "bob"],
        owner="alice",
        department="算法工程",
    )
    assert r1["status"] == "ok", f"提取失败: {r1}"
    mem_id_1 = r1["records"][0]["id"]

    ok(f"会议记忆写入成功｜ID: {c(mem_id_1, YELLOW)}")
    ok(f"自动提取决策 3 条 + 待办 3 条 + 标签 {EXTRACT_OUTPUT_1['records'][0]['tags']}")

    section("48h 后：灰度扩量决策会（alice=李明 主导），写入扩量决策")
    r2 = mem_extract_commit(
        config=config,
        llm_output=json.dumps(EXTRACT_OUTPUT_2, ensure_ascii=False),
        participants=["alice", "bob"],
        owner="alice",
        department="算法工程",
    )
    mem_id_2 = r2["records"][0]["id"]
    ok(f"扩量决策写入成功｜ID: {c(mem_id_2, YELLOW)}")

    section("写入辅助数据（人际关系、里程碑事件、文档记忆）")
    # 写入人际关系
    for rel in RELATIONSHIP_DATA:
        mem_write(
            config=config,
            mem_type="relationship",
            content=rel,
            owner=rel["person"],
            tags=[rel["department"], rel["role"]],
            visibility="public",
            department=rel["department"],
            summary=rel["summary"],
        )
    # alice=李明 / bob=王芳 / carol=张伟 / david=刘强
    ok(f"人员关系写入：李明(alice) / 王芳(bob) / 张伟(carol) / 刘强(david)（4 人）")

    # 写入里程碑事件
    for evt in EVENT_DATA:
        mem_write(
            config=config,
            mem_type="event",
            content=evt,
            owner="alice",
            participants=evt["participants"],
            tags=evt.get("tags", ["PriceEngine", "灰度"]),
            visibility="team",
            department="算法工程",
            summary=evt["summary"],
        )
    ok(f"里程碑事件写入：灰度启动 / 灰度扩量（2 条）")

    # 写入文档记忆
    for doc in DOCUMENT_DATA:
        mem_write(
            config=config,
            mem_type="document",
            content=doc,
            owner="alice",
            tags=doc["tags"],
            visibility="team",
            department="算法工程",
            summary=doc["summary"],
        )
    ok(f"文档记忆写入：评估报告 / 灰度方案文档（2 条）")

    # 记忆库统计
    status = mem_status(config)
    print()
    info(f"记忆库当前状态：共 {c(str(status['total_active']), BOLD)} 条活跃记忆")
    for t, n in status["by_type"].items():
        print(f"       {c(t, YELLOW):20s} {n} 条")

    # ══════════════════════════════════════════════════════════
    # 场景二：新人 carol=张伟 跨人检索历史灰度决策
    # ══════════════════════════════════════════════════════════
    banner("场景二：新人 carol=张伟（算法工程实习生）查询历史灰度决策")

    section("张伟 的问题（FTS 关键词检索）")
    query_carol_1 = "灰度"
    query_carol_1_display = "PriceEngine 灰度方案是什么"
    info(f"问题：{c(repr(query_carol_1_display), CYAN)}")
    info(f"检索词：{c(repr(query_carol_1), CYAN)}（提取核心词，FTS5 精确召回）")
    info(f"身份：carol=张伟（算法工程部门）")

    t0 = time.perf_counter()
    results_c1 = mem_search(
        config=config,
        query=query_carol_1,
        caller_id="carol",
        caller_dept="算法工程",
        days=90,
        top_k=5,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    ok(f"检索完成：{len(results_c1)} 条结果 | 耗时 {elapsed:.1f}ms")

    for i, r in enumerate(results_c1):
        result_item(i + 1, r, verbose)

    section("张伟 追问：灰度熔断条件是什么")
    query_carol_2 = "灰度"
    query_carol_2_display = "灰度熔断阈值是什么"
    info(f"问题：{c(repr(query_carol_2_display), CYAN)}")
    info(f"检索词：{c(repr(query_carol_2), CYAN)}（同一 query，从结果中读取决策详情）")

    results_c2 = mem_search(
        config=config,
        query=query_carol_2,
        caller_id="carol",
        caller_dept="算法工程",
        days=90,
        top_k=5,
    )
    # 从结果的 content.decisions 里过滤出熔断相关决策
    熔断_found = False
    for r in results_c2:
        if r.get("content") and isinstance(r["content"], dict):
            for dec in r["content"].get("decisions", []):
                if "熔断" in dec or "投诉率" in dec:
                    熔断_found = True
                    print(f"    {c('[命中]', GREEN)} {c(dec, BOLD)}")

    if 熔断_found:
        ok(f"从 {len(results_c2)} 条记忆中精准定位到熔断阈值决策")
    else:
        info(f"检索到 {len(results_c2)} 条灰度相关记忆（熔断条件在决策详情中，开启 --verbose 查看）")

    # ══════════════════════════════════════════════════════════
    # 场景三：carol=张伟（算法工程）用知识图谱查 PriceEngine 灰度关联人员
    # ══════════════════════════════════════════════════════════
    banner("场景三：carol=张伟（算法工程实习生）用知识图谱查 PriceEngine 灰度关联人员")

    section("图谱扩展检索：从 'PriceEngine 灰度' 出发，BFS 扩展关联实体")
    query_david = "灰度"
    info(f"查询：{c(repr(query_david), CYAN)}")
    info(f"身份：carol=张伟（算法工程部门）")
    info(f"检索模式：Hybrid + 图谱 BFS 扩展（最多 2 跳）")

    t0 = time.perf_counter()
    graph_result = mem_graph_search(
        config=config,
        query=query_david,
        caller_id="carol",
        caller_dept="算法工程",
        days=90,
        top_k=5,
        max_hops=2,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    direct = graph_result["direct"]
    expanded = graph_result["expanded"]
    entities = graph_result["entities"]
    edges = graph_result["edges"]

    ok(f"直接命中：{len(direct)} 条 | 图扩展发现：{len(expanded)} 条 | 耗时 {elapsed:.1f}ms")

    if entities:
        print(f"\n  {c('关联实体节点（知识图谱）：', BOLD)}")
        type_groups: dict = {}
        for e in entities:
            t = e.get("entity_type", "?")
            type_groups.setdefault(t, []).append(e["name"])
        for etype, names in type_groups.items():
            print(f"    {c(etype, YELLOW):12s}: {', '.join(names)}")

    if edges:
        print(f"\n  {c('关系三元组（最多显示 5 条）：', BOLD)}")
        ent_id2name = {e["id"]: e["name"] for e in entities}
        shown = 0
        for edge in edges[:5]:
            fn = ent_id2name.get(edge.get("from_entity", ""), edge.get("from_entity", "?"))
            tn = ent_id2name.get(edge.get("to_entity", ""), edge.get("to_entity", "?"))
            rt = edge.get("relation_type", "?")
            print(f"    {c(fn, CYAN)} ──{c(rt, MAGENTA)}──▶ {c(tn, CYAN)}")
            shown += 1

    section("直接命中的记忆")
    for i, r in enumerate(direct[:3]):
        result_item(i + 1, r, verbose)

    if expanded:
        section("图扩展发现的关联记忆")
        for i, r in enumerate(expanded[:2]):
            result_item(i + 1, r, verbose)

    # ══════════════════════════════════════════════════════════
    # 场景四：alice=李明 查询与 bob=王芳 的协作历史
    # ══════════════════════════════════════════════════════════
    banner("场景四：alice=李明 查询与 bob=王芳 的跨部门协作记忆链")

    section("Recall 接口：以 'bob' 为实体，检索全部关联记忆")
    info(f"查询实体：{c('bob（王芳）', CYAN)}（产品运营 PM）")
    info(f"调用方：alice=李明（算法工程）")

    t0 = time.perf_counter()
    recall_result = mem_recall(
        config=config,
        entity="bob",
        depth=20,
        caller_id="alice",
        caller_dept="算法工程",
    )
    elapsed = (time.perf_counter() - t0) * 1000

    ok(f"找到 {c(str(recall_result['total']), BOLD)} 条与 bob（王芳）相关的记忆 | 耗时 {elapsed:.1f}ms")

    for i, r in enumerate(recall_result["memories"][:4]):
        result_item(i + 1, r, verbose)

    # ══════════════════════════════════════════════════════════
    # 场景五：长程记忆验证（模拟"新会话"，carol=张伟 重新查询历史决策）
    # ══════════════════════════════════════════════════════════
    banner("场景五：长程记忆验证 — 新会话中依然能找到历史决策")

    section("模拟新会话启动（carol=张伟 完全重新开始一次对话）")
    info("传统 AI 助手在新会话中会失忆 —— OpenClaw + enterprise-memory 不会")
    info(f"张伟 在新会话中询问：{c(repr('上次和产品侧定的灰度比例是多少'), CYAN)}")
    info(f"记忆库持久化在磁盘，新会话只需重新连接同一个 DB，历史记忆全部保留")

    # 完全用新 config 对象模拟"新会话"（关键：指向同一个 DB 路径）
    new_session_config = make_config(db_path)
    # 用核心关键词"灰度"检索，FTS5 精确召回
    results_new = mem_search(
        config=new_session_config,
        query="灰度",
        caller_id="carol",
        caller_dept="算法工程",
        days=365,   # 扩大时间窗口，确保覆盖所有历史记录
        top_k=5,
    )

    if results_new:
        ok(f"即使在新会话中，依然检索到 {len(results_new)} 条历史决策！")
        for i, r in enumerate(results_new[:3]):
            result_item(i + 1, r, verbose)
            if r.get("content") and isinstance(r["content"], dict):
                for dec in r["content"].get("decisions", []):
                    if "灰度" in dec and ("1%" in dec or "5%" in dec or "比例" in dec):
                        print(f"         {c('→ 找到决策:', GREEN)} {dec}")
    else:
        warn("未能检索到历史决策（可能需要检查检索配置）")

    section("权限隔离验证：carol=张伟 无法看到 alice=李明 的私密记录")

    # 写入一条 alice 的私密记忆
    mem_write(
        config=config,
        mem_type="conversation",
        content={"summary": "alice（李明）个人备忘：下周 PriceEngine 全量发布前需要准备的评审 PPT 草稿"},
        owner="alice",
        visibility="private",
        department="算法工程",
    )

    # carol 尝试检索
    private_results = mem_search(
        config=config,
        query="alice 李明 个人备忘 PPT 草稿",
        caller_id="carol",
        caller_dept="算法工程",
        days=90,
        top_k=5,
    )
    private_leaked = any("PPT 草稿" in r.get("summary", "") for r in private_results)
    if not private_leaked:
        ok("权限隔离正常：carol=张伟 无法访问 alice=李明 的 private 记忆 ✓")
    else:
        warn("⚠️ 权限漏洞：carol=张伟 看到了 alice=李明 的私密记录！")

    # ══════════════════════════════════════════════════════════
    # 总结
    # ══════════════════════════════════════════════════════════
    banner("Demo 完成 — 核心能力验证汇总", GREEN)

    final_status = mem_status(config)
    # carol=张伟 是否通过检索找到了 alice=李明/bob=王芳 的跨部门决策
    carol_found_cross_dept = len(results_c1) > 0 and any(
        r.get("owner") in ("alice",) or
        any(p in r.get("participants", []) for p in ["alice", "bob"])
        for r in results_c1
    )

    capabilities = [
        ("跨人协作记忆",       "李明×王芳 会议决策被 张伟 检索到",    carol_found_cross_dept),
        ("自动提取",           "LLM 从对话中提取决策 & 待办",          True),
        ("知识图谱扩展",       "从 PriceEngine 实体 BFS 扩展关联人员", len(entities) > 0),
        ("人际关系 Recall",    "李明 查到与 王芳 的全部协作链路",        recall_result["total"] > 0),
        ("长程跨会话持久化",   "新会话中依然能找到历史灰度决策",        len(results_new) > 0),
        ("多用户权限隔离",     "张伟 无法看到 李明 的 private 记忆",    not private_leaked),
    ]

    print()
    for cap, desc, passed in capabilities:
        icon = c("✅", GREEN) if passed else c("❌", RED)
        status_str = c("PASS", GREEN) if passed else c("FAIL", RED)
        print(f"  {icon} {c(cap, BOLD):16s} {c(status_str, BOLD)}  {c(desc, GRAY)}")

    print()
    info(f"记忆库最终状态：共 {c(str(final_status['total_active']), BOLD)} 条活跃记忆")
    info(f"DB 路径：{c(db_path, GRAY)}")
    print()


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="enterprise-memory 端到端 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python3 demo_enterprise.py              # 使用临时 DB，Demo 结束后自动清理
  python3 demo_enterprise.py --keep-db   # 保留 DB，可用 memory_engine.py 继续查询
  python3 demo_enterprise.py --verbose   # 打印完整记忆内容（决策 & 待办详情）
  python3 demo_enterprise.py --db-path /tmp/my_demo.db  # 指定 DB 路径
        """
    )
    parser.add_argument("--keep-db",  action="store_true", help="Demo 结束后保留数据库")
    parser.add_argument("--verbose",  action="store_true", help="打印完整记忆内容")
    parser.add_argument("--db-path",  type=str, default=None, help="指定 SQLite 数据库路径")
    args = parser.parse_args()

    banner("企业级长程协作记忆引擎 — 端到端 Demo", BOLD + CYAN)
    print(f"  {c('场景', BOLD)}: 星辰电商 PriceEngine v2 智能定价系统灰度上线")
    print(f"  {c('角色', BOLD)}: alice=李明（算法工程）/ bob=王芳（产品运营 PM）/ carol=张伟（实习生）/ david=刘强（基础设施）")
    print(f"  {c('演示', BOLD)}: 自动提取 → 跨人检索 → 图谱扩展 → 长程持久化 → 权限隔离")

    # 数据库路径处理
    if args.db_path:
        db_path = args.db_path
        cleanup = False
    elif args.keep_db:
        db_path = str(Path.home() / ".openclaw/workspace/memory_engine/demo.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix="em_demo_")
        db_path = tmp.name
        tmp.close()
        cleanup = True

    info(f"使用数据库：{c(db_path, GRAY)}")
    if cleanup:
        info("Demo 结束后自动清理（使用 --keep-db 保留）")

    print()
    t_start = time.perf_counter()

    try:
        run_demo(db_path=db_path, verbose=args.verbose)
    except KeyboardInterrupt:
        print(f"\n{c('已中断', YELLOW)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{c('Demo 运行出错：', RED)}{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        elapsed = time.perf_counter() - t_start
        print(f"  {c('总耗时：', GRAY)}{elapsed:.1f}s")
        if cleanup:
            Path(db_path).unlink(missing_ok=True)
            print(f"  {c('临时数据库已清理', GRAY)}")
        else:
            print(f"  {c('数据库已保留：', GRAY)}{db_path}")
            print(f"  {c('可继续查询：', GRAY)}"
                  f"python3 scripts/memory_engine.py search --query '灰度' --caller carol")


if __name__ == "__main__":
    main()
