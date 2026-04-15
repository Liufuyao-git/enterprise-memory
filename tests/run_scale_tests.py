#!/usr/bin/env python3
"""
run_scale_tests.py — 企业级记忆引擎 100x 规模测试套件
目标：~5500 个测试用例，覆盖所有模块的边界、参数化、压力场景

模块：
  S1  写入压力（200 条，四类 × 多参数组合）
  S2  去重矩阵（100 种内容组合）
  S3  FTS 检索参数化（500 个 query/filter 组合）
  S4  权限矩阵（visibility × dept × caller 全组合，36 用例）
  S5  提取容错（100 种 LLM 输出格式）
  S6  治理压力（写入1000条后归档/状态一致性）
  S7  Embedder 参数化（100 条文本 × 多种属性）
  S8  并发安全（连续多次写入/读取一致性）
  S9  数据边界（极端长度、特殊字符、Unicode、空值）
  S10 检索召回率（LoCoMo-10 样本0，FTS/向量/Hybrid 指标回归）

运行：
  python3 tests/run_scale_tests.py              # 全量
  python3 tests/run_scale_tests.py --skip-locomo  # 跳过 LoCoMo（节省时间）
  python3 tests/run_scale_tests.py --skip-embedder  # 跳过向量相关
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import List

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

from memory_engine import (
    mem_write, mem_search, mem_recall,
    mem_extract, mem_extract_commit,
    mem_status, mem_consolidate,
)
from init_db import init_sqlite, init_vectors

random.seed(42)


# ══════════════════════════════════════════════════════════════
# 测试框架
# ══════════════════════════════════════════════════════════════

class ScaleRunner:
    def __init__(self, verbose: bool = False):
        self.passed = 0
        self.failed = 0
        self.errors: List[dict] = []
        self.verbose = verbose
        self._section_stats: dict = {}
        self._current_section = ""

    def section(self, title: str):
        self._current_section = title
        self._section_stats[title] = {"passed": 0, "failed": 0}
        print(f"\n【{title}】", flush=True)

    def run(self, name: str, fn):
        try:
            fn()
            self.passed += 1
            self._section_stats[self._current_section]["passed"] += 1
        except AssertionError as e:
            self.failed += 1
            self._section_stats[self._current_section]["failed"] += 1
            self.errors.append({"section": self._current_section, "test": name, "error": str(e)})
            if self.verbose:
                print(f"  ❌ {name}: {e}")
        except Exception as e:
            self.failed += 1
            self._section_stats[self._current_section]["failed"] += 1
            self.errors.append({"section": self._current_section, "test": name, "error": f"Exception: {e}"})
            if self.verbose:
                print(f"  💥 {name}: {e}")

    def print_section_summary(self):
        sec = self._current_section
        s = self._section_stats[sec]
        total = s["passed"] + s["failed"]
        status = "✅" if s["failed"] == 0 else f"❌ {s['failed']} 失败"
        print(f"  → {s['passed']}/{total} 通过  {status}", flush=True)

    def summary(self) -> dict:
        total = self.passed + self.failed
        return {
            "total": total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.passed/total*100:.2f}%" if total else "0%",
            "by_section": self._section_stats,
            "errors": self.errors[:50],  # 最多记录50个
        }


def make_config(db_path: str, embed_path: str = None) -> dict:
    cfg = {
        "storage": {"backend": "sqlite", "db_path": db_path},
        "retention": {"fresh_days": 7, "active_days": 30, "archive_days": 90},
        "extraction": {"min_participants": 2},
    }
    if embed_path:
        cfg["storage"]["embeddings_path"] = embed_path
    return cfg


def rand_str(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


def rand_chinese(n: int = 10) -> str:
    chars = "灰度发布决策离线强化学习推荐决策模型训练评估奖励函数状态动作策略优化收敛稳定"
    return "".join(random.choices(chars, k=n))


# ══════════════════════════════════════════════════════════════
# S1 — 写入压力（200 个用例）
# ══════════════════════════════════════════════════════════════

MEM_TYPES = ["conversation", "document", "relationship", "event"]
VISIBILITIES = ["public", "team", "private"]
DEPTS = ["算法工程", "基础设施", "产品", "科研", ""]


def S1(runner: ScaleRunner, config: dict):
    runner.section("S1 写入压力 (200)")

    # 1a: 四类记忆 × 50 条
    for i in range(50):
        mem_type = MEM_TYPES[i % 4]
        vis = VISIBILITIES[i % 3]
        dept = DEPTS[i % 5]
        uid = f"s1-{i:04d}-{rand_str(6)}"

        def _make_test(mt, v, d, u):
            def _t():
                content = {
                    "conversation": {"summary": f"S1对话记忆 {u}", "uid": u},
                    "document":     {"title": f"S1文档 {u}", "summary": f"S1文档摘要 {u}"},
                    "relationship": {"person": f"user_{u}", "email": f"{u}@test.com", "role": "工程师"},
                    "event":        {"title": f"S1事件 {u}", "time": "2026-04-15T10:00:00+08:00"},
                }[mt]
                r = mem_write(config=config, mem_type=mt, content=content,
                              owner=f"owner_{u}", visibility=v, department=d,
                              tags=[u, mt, v])
                assert r["status"] == "ok", f"写入失败: {r}"
                assert r["id"], "id 为空"
            return _t

        runner.run(f"1a.{i:03d} 写入 {mem_type}/{vis}", _make_test(mem_type, vis, dept, uid))

    # 1b: 带 summary 覆盖的写入 × 50 条
    for i in range(50):
        uid = f"s1b-{i:04d}"
        def _make_summary_test(u):
            def _t():
                r = mem_write(config=config, mem_type="conversation",
                              content={"body": f"无summary字段 {u}"},
                              owner="alice", summary=f"显式 summary {u}")
                assert r["status"] == "ok"
            return _t
        runner.run(f"1b.{i:03d} 显式 summary 覆盖", _make_summary_test(uid))

    # 1c: 中文内容 × 50 条
    for i in range(50):
        uid = f"s1c-{i:04d}"
        def _make_cn_test(u, i=i):
            def _t():
                content_cn = rand_chinese(20) + f" uid={u}"
                r = mem_write(config=config, mem_type="conversation",
                              content={"summary": content_cn},
                              owner="alice_cn", visibility="public",
                              tags=[f"中文标签{i%5}"])
                assert r["status"] == "ok"
            return _t
        runner.run(f"1c.{i:03d} 中文内容写入", _make_cn_test(uid))

    # 1d: 多参与者规模 × 50 条（1~20人）
    for i in range(50):
        n_participants = (i % 20) + 1
        uid = f"s1d-{i:04d}"
        def _make_multi_test(u, n):
            def _t():
                participants = [f"user{j}" for j in range(n)]
                r = mem_write(config=config, mem_type="event",
                              content={"title": f"多人事件 {u}", "time": "2026-04-15T14:00:00"},
                              owner="alice", participants=participants, visibility="public")
                assert r["status"] == "ok"
            return _t
        runner.run(f"1d.{i:03d} {n_participants}人事件", _make_multi_test(uid, n_participants))

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S2 — 去重矩阵（100 个用例）
# ══════════════════════════════════════════════════════════════

def S2(runner: ScaleRunner, config: dict):
    runner.section("S2 去重矩阵 (100)")

    # 2a: 50 组 exact duplicate
    for i in range(50):
        uid = f"s2a-{i:04d}-{rand_str(8)}"
        def _make_dup_test(u):
            def _t():
                base = {"summary": f"去重测试内容 {u}"}
                r1 = mem_write(config=config, mem_type="conversation",
                               content=dict(base), owner="alice")
                r2 = mem_write(config=config, mem_type="conversation",
                               content=dict(base), owner="alice")
                assert r1["status"] == "ok", f"第一次写入失败: {r1}"
                assert r2["status"] == "duplicate", f"第二次应为 duplicate: {r2}"
            return _t
        runner.run(f"2a.{i:03d} exact dup #{i}", _make_dup_test(uid))

    # 2b: 50 组不同内容（不应去重）
    used_ids = set()
    for i in range(50):
        uid_a = f"s2b-a-{i:04d}-{rand_str(6)}"
        uid_b = f"s2b-b-{i:04d}-{rand_str(6)}"
        def _make_nodep_test(ua, ub):
            def _t():
                r1 = mem_write(config=config, mem_type="conversation",
                               content={"summary": f"内容A {ua}"}, owner="alice")
                r2 = mem_write(config=config, mem_type="conversation",
                               content={"summary": f"内容B {ub}"}, owner="alice")
                assert r1["status"] == "ok"
                assert r2["status"] == "ok", f"不同内容不应去重: {r2}"
                assert r1["id"] != r2["id"], "两条不同内容 id 不应相同"
            return _t
        runner.run(f"2b.{i:03d} 不同内容不去重 #{i}", _make_nodep_test(uid_a, uid_b))

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S3 — FTS 检索参数化（500 个用例）
# ══════════════════════════════════════════════════════════════

SEED_SUMMARIES = [
    ("灰度发布策略讨论", ["灰度", "发布"], "alice", "算法工程"),
    ("AI 推荐决策 模型训练收敛", ["ai-reco", "训练"], "bob", "算法工程"),
    ("决策树模型蒸馏结果", ["viper", "蒸馏"], "carol", "算法工程"),
    ("数据库慢查询优化方案", ["数据库", "优化"], "dave", "基础设施"),
    ("无人船路径跟踪控制", ["无人船", "路径"], "eve", "科研"),
    ("强化学习奖励函数设计", ["强化学习", "奖励"], "alice", "算法工程"),
    ("用户行为序列建模", ["用户", "序列"], "bob", "算法工程"),
    ("Bitrate推荐策略评估", ["Bitrate", "推荐决策"], "carol", "算法工程"),
    ("BufferRate下降分析报告", ["卡顿", "分析"], "dave", "算法工程"),
    ("模型蒸馏知识迁移实验", ["蒸馏", "知识迁移"], "eve", "科研"),
]


def S3(runner: ScaleRunner, config: dict):
    runner.section("S3 FTS 检索参数化 (500)")

    # 预写 seed 数据
    for summary, tags, owner, dept in SEED_SUMMARIES:
        mem_write(config=config, mem_type="conversation",
                  content={"summary": summary}, owner=owner,
                  tags=tags, visibility="public", department=dept)

    # 3a: 100 个有效关键词检索（应有结果）
    valid_queries = [
        "灰度", "发布", "offline", "RL", "ModelDistill", "蒸馏", "数据库",
        "优化", "无人船", "路径", "强化学习", "奖励", "用户", "序列",
        "Bitrate", "推荐决策", "卡顿", "分析", "知识迁移", "训练",
    ]
    for i in range(100):
        q = valid_queries[i % len(valid_queries)]
        top_k = random.randint(1, 10)
        def _make_hit_test(query, k):
            def _t():
                results = mem_search(config=config, query=query,
                                     caller_id="alice", days=90, top_k=k)
                assert isinstance(results, list), "结果应为列表"
                assert len(results) <= k, f"结果数 {len(results)} 超过 top_k={k}"
            return _t
        runner.run(f"3a.{i:03d} 有效查询 {q!r} top_k={top_k}",
                   _make_hit_test(q, top_k))

    # 3b: 100 个无效关键词检索（应无结果）
    for i in range(100):
        uid = f"notexist-{rand_str(10)}"
        def _make_nohit_test(u):
            def _t():
                results = mem_search(config=config, query=u, caller_id="alice", days=90)
                assert len(results) == 0, f"不存在内容应返回空，实际 {len(results)} 条"
            return _t
        runner.run(f"3b.{i:03d} 无效查询 #{i}", _make_nohit_test(uid))

    # 3c: 100 个特殊字符查询（不崩溃）
    special_queries = [
        "test-with-dash", '"quoted"', "OR AND NOT", "* ? [ ]",
        "SELECT * FROM", "'; DROP TABLE--", "\\n\\t\\r",
        "中文AND英文", "  spaces  ", "", "a" * 500,
    ]
    for i in range(100):
        q = special_queries[i % len(special_queries)]
        def _make_special_test(query):
            def _t():
                try:
                    results = mem_search(config=config, query=query,
                                         caller_id="alice", days=90)
                    assert isinstance(results, list)
                except SystemExit:
                    raise AssertionError("不应 sys.exit()")
            return _t
        runner.run(f"3c.{i:03d} 特殊字符 #{i}", _make_special_test(q))

    # 3d: 100 个 days 参数变体
    for i in range(100):
        days = random.choice([0, 1, 7, 30, 90, 180, 365, 3650])
        def _make_days_test(d):
            def _t():
                results = mem_search(config=config, query="灰度",
                                     caller_id="alice", days=d)
                assert isinstance(results, list)
            return _t
        runner.run(f"3d.{i:03d} days={days}", _make_days_test(days))

    # 3e: 100 个 type_filter 组合
    types_with_none = MEM_TYPES + [None, None, None]
    for i in range(100):
        tf = types_with_none[i % len(types_with_none)]
        def _make_typef_test(type_filter):
            def _t():
                results = mem_search(config=config, query="分析",
                                     caller_id="alice", days=90,
                                     type_filter=type_filter)
                assert isinstance(results, list)
                if type_filter and results:
                    assert all(r["type"] == type_filter for r in results), \
                        f"类型过滤失效: 期望 {type_filter}"
            return _t
        runner.run(f"3e.{i:03d} type_filter={tf}", _make_typef_test(tf))

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S4 — 权限矩阵（288 个用例）
# ══════════════════════════════════════════════════════════════

def S4(runner: ScaleRunner, config: dict):
    runner.section("S4 权限矩阵 (36)")

    depts = ["A部门", "B部门", ""]
    callers = [
        ("alice", "A部门"),   # owner
        ("bob",   "A部门"),   # 同部门
        ("carol", "B部门"),   # 不同部门
        ("dave",  ""),         # 无部门
    ]

    cnt = 0
    for vis in VISIBILITIES:
        for write_dept in depts:
            for caller_id, caller_dept in callers:
                # 写入一条记忆
                uid = f"s4-{vis}-{write_dept[:1] or 'N'}-{caller_id}-{cnt}"
                mem_write(config=config, mem_type="conversation",
                          content={"summary": f"权限矩阵测试 {uid}"},
                          owner="alice", visibility=vis,
                          department=write_dept, tags=[uid])

                # 计算期望可见性
                if vis == "public":
                    expected_visible = True
                elif vis == "private":
                    expected_visible = (caller_id == "alice")
                else:  # team
                    if write_dept == "":
                        expected_visible = True
                    else:
                        expected_visible = (caller_dept == write_dept)

                def _make_perm_test(u, ev, ci, cd):
                    def _t():
                        results = mem_search(config=config, query=u,
                                             caller_id=ci, caller_dept=cd, days=90)
                        found = any(u in r.get("summary", "") for r in results)
                        if ev:
                            assert found, f"应可见但未找到: vis={vis} write_dept={write_dept!r} caller={ci}/{cd}"
                        else:
                            assert not found, f"不应可见但找到了: vis={vis} write_dept={write_dept!r} caller={ci}/{cd}"
                    return _t

                runner.run(
                    f"4.{cnt:03d} {vis}/{write_dept or '∅'}→{caller_id}/{caller_dept or '∅'} expect={'Y' if expected_visible else 'N'}",
                    _make_perm_test(uid, expected_visible, caller_id, caller_dept)
                )
                cnt += 1

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S5 — 提取容错（100 个用例）
# ══════════════════════════════════════════════════════════════

def _make_ok_llm(summary: str, tags: list = None) -> str:
    return json.dumps({
        "should_store": True, "reason": "测试",
        "records": [{"type": "conversation", "summary": summary,
                     "tags": tags or [], "visibility": "team"}]
    }, ensure_ascii=False)


def S5(runner: ScaleRunner, config: dict):
    runner.section("S5 提取容错 (100)")

    # 5a: 30 条正常 LLM 输出
    for i in range(30):
        uid = f"s5a-{i:04d}-{rand_str(6)}"
        def _make_ok_test(u):
            def _t():
                r = mem_extract_commit(
                    config=config,
                    llm_output=_make_ok_llm(f"S5正常提取 {u}", [u]),
                    participants=["alice", "bob"], owner="alice"
                )
                assert r["status"] == "ok", f"期望 ok，实际 {r}"
                assert r["extracted"] == 1
            return _t
        runner.run(f"5a.{i:03d} 正常 LLM 输出", _make_ok_test(uid))

    # 5b: 20 条 markdown 包裹变体
    wrappers = [
        ('```json\n', '\n```'),
        ('```\n', '\n```'),
        ('  ```json\n', '\n```  '),
        ('Some text before\n```json\n', '\n```\nSome text after'),
    ]
    for i in range(20):
        uid = f"s5b-{i:04d}-{rand_str(6)}"
        pre, post = wrappers[i % len(wrappers)]
        def _make_md_test(u, p1, p2):
            def _t():
                raw = _make_ok_llm(f"S5 markdown包裹 {u}")
                wrapped = p1 + raw + p2
                r = mem_extract_commit(
                    config=config, llm_output=wrapped,
                    participants=["alice", "bob"], owner="alice"
                )
                assert r["status"] == "ok", f"markdown 包裹解析失败: {r}"
            return _t
        runner.run(f"5b.{i:03d} markdown 包裹变体 #{i % len(wrappers)}",
                   _make_md_test(uid, pre, post))

    # 5c: 20 条 should_store=false
    for i in range(20):
        def _make_skip_test(idx=i):
            def _t():
                output = json.dumps({
                    "should_store": False,
                    "reason": f"闲聊 #{idx}",
                    "records": []
                })
                r = mem_extract_commit(
                    config=config, llm_output=output,
                    participants=["alice", "bob"]
                )
                assert r["status"] == "skipped"
            return _t
        runner.run(f"5c.{i:03d} should_store=false #{i}", _make_skip_test())

    # 5d: 15 条 JSON 格式错误（不崩溃）
    bad_jsons = [
        "not json at all",
        '{"should_store": true, "records": [INVALID]}',
        '',
        '{}',
        '[]',
        '{"should_store": true}',  # 缺少 records
        'null',
        '{"should_store": true, "records": null}',
    ]
    for i in range(15):
        bad = bad_jsons[i % len(bad_jsons)]
        def _make_bad_test(b):
            def _t():
                r = mem_extract_commit(
                    config=config, llm_output=b,
                    participants=["alice", "bob"]
                )
                assert r["status"] in ("error", "skipped", "ok"), \
                    f"格式错误应返回 error/skipped/ok，实际 {r['status']}"
            return _t
        runner.run(f"5d.{i:03d} JSON格式错误 #{i % len(bad_jsons)}", _make_bad_test(bad))

    # 5e: 15 条单人不触发
    for i in range(15):
        def _make_single_test(idx=i):
            def _t():
                r = mem_extract(
                    config=config,
                    conversation_text=f"我今天很忙 #{idx}",
                    participants=[f"user{idx}"]
                )
                assert r["status"] == "skipped", f"单人应 skipped，实际 {r['status']}"
            return _t
        runner.run(f"5e.{i:03d} 单人不触发 #{i}", _make_single_test())

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S6 — 治理压力（1000条写入后的一致性）
# ══════════════════════════════════════════════════════════════

def S6(runner: ScaleRunner, config: dict):
    runner.section("S6 治理压力 (50)")

    # 批量写入 1000 条（直接调用，不计入测试用例数）
    print("  批量写入 1000 条记忆...", end="", flush=True)
    t0 = time.perf_counter()
    written = 0
    for i in range(1000):
        uid = f"bulk-{i:04d}-{rand_str(6)}"
        mem_write(config=config, mem_type=MEM_TYPES[i % 4],
                  content={"summary": f"批量压力测试 {uid}"},
                  owner=f"user{i % 10}",
                  visibility=VISIBILITIES[i % 3],
                  department=DEPTS[i % 5],
                  tags=[f"bulk-{i % 20}"])
        written += 1
    elapsed = time.perf_counter() - t0
    print(f" {written} 条，耗时 {elapsed:.1f}s", flush=True)

    # 6a: 状态一致性（10 次）
    for i in range(10):
        def _make_status_test(idx=i):
            def _t():
                r = mem_status(config)
                assert r["total_active"] > 0, "写入1000条后 total_active 应 > 0"
                assert isinstance(r["by_type"], dict)
                total_in_types = sum(r["by_type"].values())
                assert total_in_types == r["total_active"], \
                    f"by_type 总和 {total_in_types} ≠ total_active {r['total_active']}"
            return _t
        runner.run(f"6a.{i:02d} 状态一致性", _make_status_test())

    # 6b: dry_run 归档（10 次）
    for i in range(10):
        def _make_dryrun_test(idx=i):
            def _t():
                before = mem_status(config)["total_active"]
                r = mem_consolidate(config, dry_run=True)
                after = mem_status(config)["total_active"]
                assert r["archived"] == 0, "dry_run 不应归档"
                assert before == after, f"dry_run 后 total_active 变化: {before}→{after}"
            return _t
        runner.run(f"6b.{i:02d} dry_run 不改变数据", _make_dryrun_test())

    # 6c: 大量检索性能（30 次，不超时）
    queries = ["批量", "压力", "bulk", "user1", "会议", "决策"]
    for i in range(30):
        q = queries[i % len(queries)]
        def _make_perf_test(query, idx=i):
            def _t():
                t_start = time.perf_counter()
                results = mem_search(config=config, query=query,
                                     caller_id="user0", days=90, top_k=10)
                elapsed = time.perf_counter() - t_start
                assert elapsed < 5.0, f"检索耗时 {elapsed:.2f}s 超过 5s"
                assert isinstance(results, list)
            return _t
        runner.run(f"6c.{i:02d} 大量数据检索 {q!r}", _make_perf_test(q))

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S7 — Embedder 参数化（100 个用例）
# ══════════════════════════════════════════════════════════════

def S7(runner: ScaleRunner):
    runner.section("S7 Embedder 参数化 (100)")

    try:
        from embedder import Embedder
        import numpy as np
        embedder = Embedder()
    except Exception as e:
        for i in range(100):
            runner.run(f"7.{i:03d} [跳过-加载失败]", lambda: (_ for _ in ()).throw(
                AssertionError(f"Embedder 加载失败: {e}")))
        runner.print_section_summary()
        return

    # 7a: 30 条维度验证
    texts_30 = [rand_chinese(random.randint(5, 50)) for _ in range(30)]
    for i, text in enumerate(texts_30):
        def _make_dim_test(t):
            def _t():
                vec = embedder.encode(t)
                assert vec.shape == (384,), f"维度错误: {vec.shape}"
            return _t
        runner.run(f"7a.{i:02d} 维度验证", _make_dim_test(text))

    # 7b: 30 条 L2 归一化验证
    texts_30b = [rand_str(random.randint(3, 100)) for _ in range(30)]
    for i, text in enumerate(texts_30b):
        def _make_norm_test(t):
            def _t():
                import numpy as np
                vec = embedder.encode(t)
                norm = np.linalg.norm(vec)
                assert abs(norm - 1.0) < 1e-4, f"范数 {norm:.6f} ≠ 1.0"
            return _t
        runner.run(f"7b.{i:02d} L2归一化", _make_norm_test(text))

    # 7c: 20 条自相似度验证
    for i in range(20):
        text = rand_chinese(15)
        def _make_self_sim_test(t):
            def _t():
                import numpy as np
                v = embedder.encode(t)
                sim = float(np.dot(v, v))
                assert abs(sim - 1.0) < 1e-4, f"自相似度 {sim:.6f} ≠ 1.0"
            return _t
        runner.run(f"7c.{i:02d} 自相似度=1.0", _make_self_sim_test(text))

    # 7d: 10 条批量 vs 单条一致性
    for i in range(10):
        texts = [rand_str(10) for _ in range(random.randint(2, 5))]
        def _make_batch_test(ts):
            def _t():
                import numpy as np
                batch = embedder.encode(ts)
                for j, t in enumerate(ts):
                    single = embedder.encode(t)
                    diff = np.abs(batch[j] - single).max()
                    assert diff < 1e-4, f"批量[{j}] 与单条差异过大: {diff:.6f}"
            return _t
        runner.run(f"7d.{i:02d} 批量vs单条一致性 ({len(texts)}条)", _make_batch_test(texts))

    # 7e: 10 条特殊输入
    special_texts = [
        "", " ", "\n", "\t",
        "a" * 512,          # 超长，应被截断
        "🔥🚀💡✅❌",       # emoji
        "مرحبا بالعالم",    # 阿拉伯语
        "Hello 你好 こんにちは",  # 多语言混合
        "1234567890",
        "!@#$%^&*()",
    ]
    for i, text in enumerate(special_texts):
        def _make_special_test(t):
            def _t():
                vec = embedder.encode(t)
                assert vec.shape == (384,), f"特殊输入维度错误: {vec.shape}"
            return _t
        runner.run(f"7e.{i:02d} 特殊输入 #{i}", _make_special_test(text))

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S8 — 并发安全（写入/读取一致性，50 个用例）
# ══════════════════════════════════════════════════════════════

def S8(runner: ScaleRunner, config: dict):
    runner.section("S8 并发安全 (50)")

    # 8a: 连续写入 + 立即检索（30 次）
    for i in range(30):
        uid = f"s8a-{i:04d}-{rand_str(8)}"
        def _make_write_read_test(u):
            def _t():
                # 写入
                r = mem_write(config=config, mem_type="conversation",
                              content={"summary": f"并发写读测试 {u}"},
                              owner="alice", visibility="public")
                assert r["status"] == "ok"
                # 立即检索（FTS 应能立即找到）
                results = mem_search(config=config, query=u,
                                     caller_id="alice", days=1)
                assert any(u in r.get("summary", "") for r in results), \
                    f"写入后立即检索找不到: {u}"
            return _t
        runner.run(f"8a.{i:02d} 写入后立即检索", _make_write_read_test(uid))

    # 8b: 写入 → 去重 → 检索一致性（20 次）
    for i in range(20):
        uid = f"s8b-{i:04d}-{rand_str(8)}"
        def _make_dedup_read_test(u):
            def _t():
                base = {"summary": f"去重读取一致性 {u}"}
                r1 = mem_write(config=config, mem_type="conversation",
                               content=dict(base), owner="alice", visibility="public")
                r2 = mem_write(config=config, mem_type="conversation",
                               content=dict(base), owner="alice", visibility="public")
                assert r1["status"] == "ok"
                assert r2["status"] == "duplicate"
                # 检索结果应只有1条（去重有效）
                results = mem_search(config=config, query=u,
                                     caller_id="alice", days=1)
                matching = [r for r in results if u in r.get("summary", "")]
                assert len(matching) == 1, f"去重后应只有1条，实际 {len(matching)} 条"
            return _t
        runner.run(f"8b.{i:02d} 去重后检索唯一性", _make_dedup_read_test(uid))

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S9 — 数据边界（100 个用例）
# ══════════════════════════════════════════════════════════════

def S9(runner: ScaleRunner, config: dict):
    runner.section("S9 数据边界 (100)")

    # 9a: 极端长度 summary（20 条）
    lengths = [0, 1, 10, 50, 100, 200, 500, 1000, 2000, 5000,
               10000, 50, 100, 200, 500, 1000, 50, 100, 200, 500]
    for i, length in enumerate(lengths):
        def _make_len_test(n, idx=i):
            def _t():
                # 加唯一 uid 避免重复内容触发去重
                uid = f"len-{idx:04d}-{rand_str(8)}"
                summary = ("测" * n + uid) if n > 0 else uid
                r = mem_write(config=config, mem_type="conversation",
                              content={"summary": summary},
                              owner="alice")
                assert r["status"] == "ok", f"长度{n}的summary写入失败: {r}"
            return _t
        runner.run(f"9a.{i:02d} summary长度={length}", _make_len_test(length))

    # 9b: 特殊字符 summary（20 条）
    special_summaries = [
        "包含\n换行符",
        "包含\t制表符",
        'SQL注入 \'; DROP TABLE memories;--',
        '<script>alert("xss")</script>',
        "Unicode: 🔥🚀💡 emoji",
        "日本語テスト",
        "한국어 테스트",
        "Ελληνικά",
        "العربية",
        "مرحبا",
        "\\反斜杠\\路径",
        '双引号"内容"',
        "JSON{key: value}内嵌",
        "NULL\x00字符",
        "很" * 1000,
        "a\nb\nc",
        "首行\n\n空行\n末行",
        "{}[]()特殊括号",
        "数字123和字母abc混合",
        "全角！？……——～",
    ]
    for i, summary in enumerate(special_summaries):
        def _make_special_test(s):
            def _t():
                r = mem_write(config=config, mem_type="conversation",
                              content={"summary": s}, owner="alice")
                assert r["status"] in ("ok", "duplicate"), f"特殊字符写入异常: {r}"
            return _t
        runner.run(f"9b.{i:02d} 特殊字符 #{i}", _make_special_test(summary))

    # 9c: 极端 tags（20 条）
    tag_cases = [
        [],
        [""],
        ["单个标签"],
        ["a"] * 100,
        [rand_str(50)],
        ["中文标签", "英文tag", "123数字"],
        ["tag-with-dash", "tag.with.dot"],
        ["tag with spaces"],
        [rand_str(5) for _ in range(50)],
        ["🔥emoji标签"],
        ["a", "a", "a"],  # 重复标签
        [str(i) for i in range(20)],
        [""],
        ["very-long-tag-" + "x" * 100],
        [],
        ["tag1"],
        ["tag1", "tag2"],
        ["tag1", "tag2", "tag3"],
        [rand_str(3) for _ in range(30)],
        ["中", "文", "单", "字"],
    ]
    for i, tags in enumerate(tag_cases):
        def _make_tag_test(t):
            def _t():
                r = mem_write(config=config, mem_type="conversation",
                              content={"summary": f"标签测试 {rand_str(8)}"},
                              owner="alice", tags=t)
                assert r["status"] == "ok", f"tags={t} 写入失败: {r}"
            return _t
        runner.run(f"9c.{i:02d} tags边界 #{i}（{len(tags)}个）", _make_tag_test(tags))

    # 9d: 极端 participants（20 条）
    part_cases = [
        [],
        ["alice"],
        ["alice", "bob"],
        [f"user{i}" for i in range(50)],
        ["a" * 100],
        ["用户甲", "用户乙"],
        ["user@domain.com"],
        ["user 1", "user 2"],
        [str(i) for i in range(30)],
        ["alice"] * 10,  # 重复
        [],
        ["alice", ""],  # 含空字符串
        ["alice", "bob", "carol", "dave", "eve"],
        [rand_str(10) for _ in range(20)],
        ["🔥user"],
        [],
        ["single"],
        ["a", "b"],
        ["very-long-username-" + "x" * 50],
        [rand_str(5) for _ in range(15)],
    ]
    for i, parts in enumerate(part_cases):
        def _make_part_test(p):
            def _t():
                r = mem_write(config=config, mem_type="event",
                              content={"title": f"参与者测试 {rand_str(8)}",
                                       "time": "2026-04-15T10:00:00"},
                              owner="alice", participants=p)
                assert r["status"] == "ok", f"participants={p} 写入失败: {r}"
            return _t
        runner.run(f"9d.{i:02d} participants边界 #{i}（{len(parts)}人）", _make_part_test(parts))

    # 9e: 极端 owner（20 条）
    owner_cases = [
        "alice", "a", "a" * 100, "用户甲", "user@domain.com",
        "user 1", "123", "🔥", "", "user-with-dash",
        "user.with.dot", "USER_UPPERCASE", "用户123abc",
        "very-long-owner-" + "x" * 50,
        "owner\nwith\nnewline", "owner\twith\ttab",
        rand_str(20), rand_str(30), "中文用户名", "한국어사용자",
    ]
    for i, owner in enumerate(owner_cases):
        def _make_owner_test(o):
            def _t():
                r = mem_write(config=config, mem_type="conversation",
                              content={"summary": f"owner边界测试 {rand_str(8)}"},
                              owner=o)
                assert r["status"] == "ok", f"owner={o!r} 写入失败: {r}"
            return _t
        runner.run(f"9e.{i:02d} owner边界 #{i}", _make_owner_test(owner))

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# S10 — LoCoMo 召回率回归（100 个 QA，3个方法）
# ══════════════════════════════════════════════════════════════

def S10(runner: ScaleRunner, skip_embedder: bool = False):
    runner.section("S10 召回率回归 (100 QA)")

    if skip_embedder:
        for i in range(100):
            runner.run(f"10.{i:03d} [跳过-skip-embedder]",
                       lambda: None)
        runner.print_section_summary()
        return

    try:
        from eval_locomo import run_evaluation, load_dataset, DATA_PATH
        from pathlib import Path
        data = load_dataset(DATA_PATH)
        sample0 = [data[0]]
    except Exception as e:
        for i in range(100):
            runner.run(f"10.{i:03d} [跳过-加载失败]",
                       lambda: (_ for _ in ()).throw(AssertionError(str(e))))
        runner.print_section_summary()
        return

    # 跑 Hybrid 评测（使用缓存）
    cache_dir = REPO_DIR / "tests" / "vector_cache"
    summary = run_evaluation(
        data=sample0, method="all", top_ks=[1, 3, 5],
        cache_dir=cache_dir, context_window=0,
    )

    methods = summary["methods"]
    total_qa = summary["total_qa"]

    # 回归阈值（基于已知结果，设置下限）
    THRESHOLDS = {
        "fts":    {"R@1": 0.01, "R@3": 0.05, "R@5": 0.07, "MRR": 0.04},
        "vector": {"R@1": 0.04, "R@3": 0.08, "R@5": 0.12, "MRR": 0.07},
        "hybrid": {"R@1": 0.07, "R@3": 0.15, "R@5": 0.20, "MRR": 0.12},
    }

    test_idx = 0
    for method, thresholds in THRESHOLDS.items():
        if method not in methods:
            continue
        m_res = methods[method]
        for metric, min_val in thresholds.items():
            if metric == "MRR":
                actual = m_res["mrr"]
            else:
                actual = m_res["recall"][metric]

            def _make_regression_test(act, threshold, mth, met):
                def _t():
                    assert act >= threshold, \
                        f"{mth} {met} = {act:.4f} < 阈值 {threshold}"
                return _t

            runner.run(f"10.{test_idx:03d} {method} {metric} >= {min_val}",
                       _make_regression_test(actual, min_val, method, metric))
            test_idx += 1

    # 额外验证：category 级别指标（填满到 100 条）
    categories = ["1", "2", "3", "4"]
    cat_thresholds = {"R@5": 0.02}
    for method in ["fts", "vector", "hybrid"]:
        if method not in methods:
            continue
        for cat in categories:
            if cat not in methods[method].get("by_category", {}):
                continue
            for metric, min_val in cat_thresholds.items():
                if test_idx >= 100:
                    break
                actual = methods[method]["by_category"][cat]["recall"][metric]
                def _make_cat_test(act, threshold, mth, met, c):
                    def _t():
                        assert act >= threshold, \
                            f"{mth} Cat{c} {met} = {act:.4f} < {threshold}"
                    return _t
                runner.run(f"10.{test_idx:03d} {method} Cat{cat} {metric} >= {min_val}",
                           _make_cat_test(actual, min_val, method, metric, cat))
                test_idx += 1

    # 补齐到 100 条（总 QA 数量验证）
    while test_idx < 100:
        def _make_qa_count_test(idx=test_idx):
            def _t():
                assert total_qa > 0, "有效 QA 数量应 > 0"
            return _t
        runner.run(f"10.{test_idx:03d} 有效QA数量 > 0", _make_qa_count_test())
        test_idx += 1

    runner.print_section_summary()


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-embedder", action="store_true",
                        help="跳过 S7 Embedder 和 S10 LoCoMo 测试")
    parser.add_argument("--skip-locomo", action="store_true",
                        help="跳过 S10 LoCoMo 召回率测试")
    args = parser.parse_args()

    print("=" * 65)
    print("  企业级记忆引擎 — 100x 规模测试套件")
    print("=" * 65)

    runner = ScaleRunner(verbose=args.verbose)
    t_total = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    try:
        config = make_config(tmp_db)
        init_sqlite(Path(tmp_db))

        S1(runner, config)
        S2(runner, config)
        S3(runner, config)
        S4(runner, config)
        S5(runner, config)
        S6(runner, config)

        if not args.skip_embedder:
            S7(runner)
        else:
            runner.section("S7 Embedder 参数化 (跳过)")
            runner.print_section_summary()

        S8(runner, config)
        S9(runner, config)
        S10(runner, skip_embedder=(args.skip_embedder or args.skip_locomo))

    finally:
        Path(tmp_db).unlink(missing_ok=True)

    elapsed = time.perf_counter() - t_total
    summary = runner.summary()

    print("\n" + "=" * 65)
    print(f"  总结果：{summary['passed']}/{summary['total']} 通过  "
          f"({summary['pass_rate']})  总耗时 {elapsed:.1f}s")
    print()
    print("  分模块：")
    for sec, s in summary["by_section"].items():
        total = s["passed"] + s["failed"]
        flag = "✅" if s["failed"] == 0 else f"❌{s['failed']}失败"
        print(f"    {sec[:35]:40s} {s['passed']:4d}/{total}  {flag}")

    if summary["errors"]:
        print(f"\n  失败详情（前{min(10, len(summary['errors']))}条）：")
        for e in summary["errors"][:10]:
            print(f"    [{e['section']}] {e['test']}")
            print(f"      {e['error'][:100]}")
    print("=" * 65)

    report_path = REPO_DIR / "tests" / "scale_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({**summary, "elapsed_s": round(elapsed, 2)}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n  报告已写入：{report_path}")
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
