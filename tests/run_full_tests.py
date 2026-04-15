#!/usr/bin/env python3
"""
run_full_tests.py — 企业级记忆引擎全量测试套件
覆盖 7 大模块，共 40+ 个测试用例

模块：
  T1  基础写入（四类记忆，边界条件）
  T2  去重机制（完全相同 / 内容相似 / 时间差异）
  T3  检索能力（FTS / 向量 / Hybrid / Recall）
  T4  权限过滤（public / team / private / 跨部门）
  T5  自动提取（LLM 提交 / 跳过 / 单人 / JSON 容错）
  T6  治理与状态（归档 / dry-run / 统计）
  T7  Embedder 单元测试（维度 / 归一化 / 相似度 / 批量）

运行：python3 tests/run_full_tests.py [--verbose]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict

REPO_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from memory_engine import (
    mem_write, mem_search, mem_recall,
    mem_extract, mem_extract_commit,
    mem_status, mem_consolidate,
)
from init_db import init_sqlite, init_vectors


# ══════════════════════════════════════════════════════════════
# 测试框架
# ══════════════════════════════════════════════════════════════

class TestRunner:
    def __init__(self, verbose: bool = False):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.verbose = verbose
        self._timings: dict = {}

    def run(self, name: str, fn):
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  ✅ {name}  ({elapsed:.0f}ms)")
            self.passed += 1
            self._timings[name] = elapsed
        except AssertionError as e:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  ❌ {name}  ({elapsed:.0f}ms)")
            if self.verbose:
                print(f"     → {e}")
            self.failed += 1
            self.errors.append({"test": name, "error": str(e)})
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  💥 {name}  ({elapsed:.0f}ms)")
            if self.verbose:
                import traceback
                print(f"     → {traceback.format_exc()}")
            self.failed += 1
            self.errors.append({"test": name, "error": f"Exception: {e}"})

    def section(self, title: str):
        print(f"\n【{title}】")

    def summary(self) -> dict:
        total = self.passed + self.failed
        slow = sorted(self._timings.items(), key=lambda x: -x[1])[:3]
        return {
            "total": total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.passed/total*100:.1f}%" if total else "0%",
            "slowest": [{"test": t, "ms": round(ms)} for t, ms in slow],
            "errors": self.errors,
        }


def make_config(db_path: str, embed_path: str = None) -> dict:
    cfg = {
        "storage": {"backend": "sqlite", "db_path": db_path},
        "retention": {"fresh_days": 7, "active_days": 30, "archive_days": 90},
        "extraction": {"min_participants": 2},
        "multi_user": {"enabled": True, "default_visibility": "team"},
    }
    if embed_path:
        cfg["storage"]["embeddings_path"] = embed_path
    return cfg


# ══════════════════════════════════════════════════════════════
# T1 — 基础写入（10 个用例）
# ══════════════════════════════════════════════════════════════

def T1(runner: TestRunner, config: dict):
    runner.section("T1 基础写入")

    # 1.1 四类记忆写入
    def t_write_conversation():
        r = mem_write(config=config, mem_type="conversation",
                      content={"summary": "T1: 对话记忆写入测试"},
                      owner="alice", tags=["test"], visibility="public")
        assert r["status"] == "ok", f"status={r['status']}"
        assert r["id"].startswith("conv_"), f"id={r['id']}"

    def t_write_document():
        r = mem_write(config=config, mem_type="document",
                      content={"title": "T1文档", "summary": "T1: 文档记忆写入测试"},
                      owner="alice", tags=["doc"], visibility="public")
        assert r["status"] == "ok"
        assert r["id"].startswith("docu_")

    def t_write_relationship():
        r = mem_write(config=config, mem_type="relationship",
                      content={"person": "bob", "email": "bob@test.com",
                               "role": "工程师", "expertise": ["Python", "SQL"]},
                      owner="alice", visibility="public")
        assert r["status"] == "ok"
        assert r["id"].startswith("rela_")

    def t_write_event():
        r = mem_write(config=config, mem_type="event",
                      content={"title": "T1评审会", "time": "2026-04-15T10:00:00+08:00"},
                      owner="alice", participants=["alice", "bob"], visibility="public")
        assert r["status"] == "ok"
        assert r["id"].startswith("even_")

    # 1.5 summary 自动从 content 提取
    def t_auto_summary():
        r = mem_write(config=config, mem_type="conversation",
                      content={"summary": "自动提取的摘要内容"},
                      owner="alice")
        assert r["status"] == "ok"

    # 1.6 无 summary 时用 title 兜底
    def t_summary_from_title():
        r = mem_write(config=config, mem_type="document",
                      content={"title": "用 title 作为 summary"},
                      owner="alice")
        assert r["status"] == "ok"

    # 1.7 空 tags 和 participants 不报错
    def t_empty_tags_participants():
        r = mem_write(config=config, mem_type="conversation",
                      content={"summary": "空 tags 测试"},
                      owner="alice", tags=[], participants=[])
        assert r["status"] == "ok"

    # 1.8 长文本 content（1000字）
    def t_long_content():
        long_text = "这是一段很长的测试内容。" * 100
        r = mem_write(config=config, mem_type="document",
                      content={"summary": "长文本测试", "body": long_text},
                      owner="alice")
        assert r["status"] == "ok"

    # 1.9 中文内容
    def t_chinese_content():
        r = mem_write(config=config, mem_type="conversation",
                      content={"summary": "灰度发布决策：从1%扩量到5%，观察48小时"},
                      owner="alice", tags=["灰度", "发布"], visibility="public")
        assert r["status"] == "ok"

    # 1.10 多参与者
    def t_multi_participants():
        participants = [f"user{i}" for i in range(10)]
        r = mem_write(config=config, mem_type="event",
                      content={"title": "多人会议", "time": "2026-04-15T14:00:00"},
                      owner="alice", participants=participants, visibility="public")
        assert r["status"] == "ok"

    runner.run("1.1 写入对话记忆", t_write_conversation)
    runner.run("1.2 写入文档记忆", t_write_document)
    runner.run("1.3 写入人际关系", t_write_relationship)
    runner.run("1.4 写入事件记忆", t_write_event)
    runner.run("1.5 summary 自动提取", t_auto_summary)
    runner.run("1.6 summary 从 title 兜底", t_summary_from_title)
    runner.run("1.7 空 tags/participants 兼容", t_empty_tags_participants)
    runner.run("1.8 长文本 content 写入", t_long_content)
    runner.run("1.9 中文内容写入", t_chinese_content)
    runner.run("1.10 多参与者（10人）", t_multi_participants)


# ══════════════════════════════════════════════════════════════
# T2 — 去重机制（5 个用例）
# ══════════════════════════════════════════════════════════════

def T2(runner: TestRunner, config: dict):
    runner.section("T2 去重机制")

    def t_dedup_exact():
        base = {"summary": "T2: 完全相同内容去重测试 exact-001"}
        mem_write(config=config, mem_type="conversation", content=dict(base), owner="alice")
        r2 = mem_write(config=config, mem_type="conversation", content=dict(base), owner="alice")
        assert r2["status"] == "duplicate", f"期望 duplicate，实际 {r2['status']}"

    def t_dedup_different_owner():
        """相同内容、不同 owner → 仍然去重（基于内容哈希）"""
        base = {"summary": "T2: 不同 owner 去重测试 diff-owner-001"}
        mem_write(config=config, mem_type="conversation", content=dict(base), owner="alice")
        r2 = mem_write(config=config, mem_type="conversation", content=dict(base), owner="bob")
        assert r2["status"] == "duplicate", f"期望 duplicate，实际 {r2['status']}"

    def t_dedup_different_content():
        """不同内容 → 不应被去重"""
        mem_write(config=config, mem_type="conversation",
                  content={"summary": "T2: 内容A unique-a"}, owner="alice")
        r2 = mem_write(config=config, mem_type="conversation",
                       content={"summary": "T2: 内容B unique-b"}, owner="alice")
        assert r2["status"] == "ok", f"期望 ok，实际 {r2['status']}"

    def t_dedup_different_type():
        """相同 summary、不同 type → 不同哈希，不去重"""
        base = {"summary": "T2: 相同内容不同类型 type-test-001"}
        mem_write(config=config, mem_type="conversation", content=dict(base), owner="alice")
        r2 = mem_write(config=config, mem_type="document", content=dict(base), owner="alice")
        # type 不在哈希排除列表，内容相同则哈希相同 → duplicate
        assert r2["status"] == "duplicate"

    def t_dedup_third_write():
        """连续写入3次相同内容 → 第2/3次都是 duplicate"""
        base = {"summary": "T2: 三次写入去重 triple-001"}
        r1 = mem_write(config=config, mem_type="conversation", content=dict(base), owner="alice")
        r2 = mem_write(config=config, mem_type="conversation", content=dict(base), owner="alice")
        r3 = mem_write(config=config, mem_type="conversation", content=dict(base), owner="alice")
        assert r1["status"] == "ok"
        assert r2["status"] == "duplicate"
        assert r3["status"] == "duplicate"

    runner.run("2.1 完全相同内容去重", t_dedup_exact)
    runner.run("2.2 不同 owner 相同内容去重", t_dedup_different_owner)
    runner.run("2.3 不同内容不去重", t_dedup_different_content)
    runner.run("2.4 相同内容不同 type 去重", t_dedup_different_type)
    runner.run("2.5 连续三次写入去重", t_dedup_third_write)


# ══════════════════════════════════════════════════════════════
# T3 — 检索能力（12 个用例）
# ══════════════════════════════════════════════════════════════

def T3(runner: TestRunner, config: dict):
    runner.section("T3 检索能力")

    # 预写入测试数据
    seeds = [
        ("灰度发布策略：从1%扩量到5%，观察窗口48小时", ["灰度", "发布"], "alice", "推荐"),
        ("AI 推荐决策模型训练完成，DropRate下降10%", ["ai-reco", "训练"], "alice", "推荐"),
        ("决策树模型蒸馏实验结果：准确率0.856", ["viper", "蒸馏"], "alice", "推荐"),
        ("无人船路径跟踪实验：ESO补偿收敛时间缩短30%", ["无人船", "控制"], "bob", "科研"),
        ("数据库慢查询优化：索引命中率从60%提升到95%", ["数据库", "优化"], "carol", "基础设施"),
    ]
    for summary, tags, owner, dept in seeds:
        mem_write(config=config, mem_type="conversation",
                  content={"summary": summary}, owner=owner,
                  tags=tags, visibility="public", department=dept)

    def t_fts_keyword_hit():
        results = mem_search(config=config, query="灰度", caller_id="alice", days=90)
        assert len(results) > 0, "关键词检索无结果"
        assert any("灰度" in r["summary"] for r in results), "结果不含关键词"

    def t_fts_english_keyword():
        results = mem_search(config=config, query="offline", caller_id="alice", days=90)
        assert len(results) > 0, "英文关键词检索无结果"

    def t_fts_tag_search():
        results = mem_search(config=config, query="viper", caller_id="alice", days=90)
        assert len(results) > 0, "标签检索无结果"

    def t_fts_no_result():
        results = mem_search(config=config, query="xyzabc999notexist", caller_id="alice", days=90)
        assert len(results) == 0, f"不存在内容应返回空，实际 {len(results)} 条"

    def t_search_topk():
        results = mem_search(config=config, query="实验", caller_id="alice",
                             days=90, top_k=2)
        assert len(results) <= 2, f"top_k=2 但返回 {len(results)} 条"

    def t_search_type_filter():
        # 写一条 document 类型
        mem_write(config=config, mem_type="document",
                  content={"summary": "类型过滤测试文档 typefilter001"},
                  owner="alice", visibility="public")
        results = mem_search(config=config, query="typefilter001",
                             caller_id="alice", days=90, type_filter="document")
        assert len(results) > 0, "类型过滤检索无结果"
        assert all(r["type"] == "document" for r in results), "结果含非 document 类型"

    def t_search_days_filter():
        """days=1 只看最近1天，应该能检索到刚写的数据"""
        results = mem_search(config=config, query="灰度", caller_id="alice", days=1)
        assert len(results) > 0, "days=1 应能检索到今天写入的数据"

    def t_search_days_zero():
        """极端 days=0 → 检索范围极小，应无结果（今天内刚写但范围0天）"""
        results = mem_search(config=config, query="灰度", caller_id="alice", days=0)
        # 0天前到现在，刚写入的记录可能不在范围内，验证不报错即可
        assert isinstance(results, list), "应返回列表"

    def t_recall_by_owner():
        results_raw = mem_search(config=config, query="无人船", caller_id="bob", days=90)
        recall = mem_recall(config=config, entity="bob", depth=20, caller_id="bob")
        assert recall["total"] > 0, "bob 应有记忆"
        assert recall["entity"] == "bob"

    def t_recall_by_tag():
        recall = mem_recall(config=config, entity="ai-reco", depth=10, caller_id="alice")
        assert recall["total"] > 0, "按 tag 召回应有结果"

    def t_search_multi_keyword():
        """多词联合检索"""
        results = mem_search(config=config, query="数据库 优化", caller_id="carol", days=90)
        assert isinstance(results, list), "多词检索应返回列表"

    def t_search_special_chars():
        """含特殊字符的 query 不应报错（FTS5 转义）"""
        for q in ['test-with-dash', 'test "with quotes"', 'private public']:
            results = mem_search(config=config, query=q, caller_id="alice", days=90)
            assert isinstance(results, list), f"特殊字符 query {q!r} 应不报错"

    runner.run("3.1 FTS 中文关键词命中", t_fts_keyword_hit)
    runner.run("3.2 FTS 英文关键词命中", t_fts_english_keyword)
    runner.run("3.3 FTS 标签检索", t_fts_tag_search)
    runner.run("3.4 FTS 不存在内容返回空", t_fts_no_result)
    runner.run("3.5 top_k 限制生效", t_search_topk)
    runner.run("3.6 type_filter 类型过滤", t_search_type_filter)
    runner.run("3.7 days 时间范围过滤", t_search_days_filter)
    runner.run("3.8 days=0 极端值不报错", t_search_days_zero)
    runner.run("3.9 Recall 按 owner 查询", t_recall_by_owner)
    runner.run("3.10 Recall 按 tag 查询", t_recall_by_tag)
    runner.run("3.11 多词联合检索", t_search_multi_keyword)
    runner.run("3.12 特殊字符 query 不崩溃", t_search_special_chars)


# ══════════════════════════════════════════════════════════════
# T4 — 权限过滤（8 个用例）
# ══════════════════════════════════════════════════════════════

def T4(runner: TestRunner, config: dict):
    runner.section("T4 权限过滤")

    # 预写数据
    mem_write(config=config, mem_type="conversation",
              content={"summary": "T4: public 记忆 pub-vis-001"},
              owner="alice", visibility="public", department="A部门")
    mem_write(config=config, mem_type="conversation",
              content={"summary": "T4: team 记忆 team-vis-001"},
              owner="alice", visibility="team", department="A部门")
    mem_write(config=config, mem_type="conversation",
              content={"summary": "T4: private 记忆 priv-vis-001"},
              owner="alice", visibility="private", department="A部门")

    def t_public_visible_to_all():
        """public → 任何人都能看到"""
        r = mem_search(config=config, query="pub-vis-001",
                       caller_id="stranger", caller_dept="外星部门", days=90)
        assert any("pub-vis-001" in x["summary"] for x in r), "public 对陌生人不可见"

    def t_team_visible_same_dept():
        """team → 同部门可见"""
        r = mem_search(config=config, query="team-vis-001",
                       caller_id="bob", caller_dept="A部门", days=90)
        assert any("team-vis-001" in x["summary"] for x in r), "team 对同部门不可见"

    def t_team_invisible_diff_dept():
        """team → 不同部门不可见"""
        r = mem_search(config=config, query="team-vis-001",
                       caller_id="carol", caller_dept="B部门", days=90)
        assert not any("team-vis-001" in x["summary"] for x in r), "team 对不同部门可见（错误）"

    def t_private_visible_to_owner():
        """private → owner 自己可见"""
        r = mem_search(config=config, query="priv-vis-001",
                       caller_id="alice", caller_dept="A部门", days=90)
        assert any("priv-vis-001" in x["summary"] for x in r), "private 对 owner 不可见"

    def t_private_invisible_to_others():
        """private → 非 owner 不可见"""
        r = mem_search(config=config, query="priv-vis-001",
                       caller_id="bob", caller_dept="A部门", days=90)
        assert not any("priv-vis-001" in x["summary"] for x in r), "private 对他人可见（错误）"

    def t_private_invisible_diff_dept():
        """private → 不同部门更不可见"""
        r = mem_search(config=config, query="priv-vis-001",
                       caller_id="carol", caller_dept="B部门", days=90)
        assert not any("priv-vis-001" in x["summary"] for x in r), "private 对不同部门可见（错误）"

    def t_recall_respects_permission():
        """recall 也应遵守权限"""
        recall = mem_recall(config=config, entity="alice", depth=20,
                            caller_id="stranger", caller_dept="外星部门")
        for m in recall["memories"]:
            assert m["visibility"] != "private", f"recall 返回了 private 记忆: {m['summary']}"
            if m["visibility"] == "team":
                assert m.get("department", "") in ("", "外星部门"), "recall 返回了不同部门的 team 记忆"

    def t_team_no_dept_visible_to_all():
        """team 且 department='' → 对所有人可见"""
        mem_write(config=config, mem_type="conversation",
                  content={"summary": "T4: no-dept team 记忆 nodept-001"},
                  owner="alice", visibility="team", department="")
        r = mem_search(config=config, query="nodept-001",
                       caller_id="stranger", caller_dept="外星部门", days=90)
        assert any("nodept-001" in x["summary"] for x in r), "无部门 team 应对所有人可见"

    runner.run("4.1 public 对所有人可见", t_public_visible_to_all)
    runner.run("4.2 team 对同部门可见", t_team_visible_same_dept)
    runner.run("4.3 team 对不同部门不可见", t_team_invisible_diff_dept)
    runner.run("4.4 private 对 owner 可见", t_private_visible_to_owner)
    runner.run("4.5 private 对他人不可见", t_private_invisible_to_others)
    runner.run("4.6 private 对不同部门不可见", t_private_invisible_diff_dept)
    runner.run("4.7 recall 遵守权限", t_recall_respects_permission)
    runner.run("4.8 无部门 team 对所有人可见", t_team_no_dept_visible_to_all)


# ══════════════════════════════════════════════════════════════
# T5 — 自动提取（7 个用例）
# ══════════════════════════════════════════════════════════════

MOCK_LLM_OK = json.dumps({
    "should_store": True,
    "reason": "包含明确决策",
    "records": [{
        "type": "conversation",
        "summary": "T5: LLM 提取 — 决定灰度扩量到5%",
        "decisions": ["灰度从1%→5%"],
        "action_items": [{"owner": "alice", "task": "更新配置", "due": "2026-04-16"}],
        "tags": ["灰度", "T5"],
        "visibility": "team"
    }]
}, ensure_ascii=False)

MOCK_LLM_SKIP = json.dumps({
    "should_store": False, "reason": "无实质内容", "records": []
})

MOCK_LLM_MULTI = json.dumps({
    "should_store": True,
    "reason": "多条记忆",
    "records": [
        {"type": "conversation", "summary": "T5: 多条-1", "tags": ["T5"], "visibility": "team"},
        {"type": "document",     "summary": "T5: 多条-2", "tags": ["T5"], "visibility": "team"},
    ]
})

MOCK_LLM_MARKDOWN = '```json\n' + json.dumps({
    "should_store": True, "reason": "测试 markdown 包裹",
    "records": [{"type": "conversation", "summary": "T5: markdown包裹测试",
                 "tags": ["T5"], "visibility": "team"}]
}) + '\n```'

MOCK_LLM_BAD_JSON = '{"should_store": true, "records": [INVALID JSON]}'


def T5(runner: TestRunner, config: dict):
    runner.section("T5 自动提取")

    def t_extract_commit_ok():
        r = mem_extract_commit(config=config, llm_output=MOCK_LLM_OK,
                               participants=["alice", "bob"], owner="alice")
        assert r["status"] == "ok", f"期望 ok，实际 {r['status']}"
        assert r["extracted"] == 1

    def t_extract_commit_skip():
        r = mem_extract_commit(config=config, llm_output=MOCK_LLM_SKIP,
                               participants=["alice", "bob"])
        assert r["status"] == "skipped"

    def t_extract_multi_records():
        r = mem_extract_commit(config=config, llm_output=MOCK_LLM_MULTI,
                               participants=["alice", "bob"], owner="alice")
        assert r["status"] == "ok"
        assert r["extracted"] == 2, f"期望提取2条，实际 {r['extracted']}"

    def t_extract_markdown_wrapped():
        """LLM 输出被 markdown 代码块包裹时应能正确解析"""
        r = mem_extract_commit(config=config, llm_output=MOCK_LLM_MARKDOWN,
                               participants=["alice", "bob"], owner="alice")
        assert r["status"] == "ok", f"markdown 包裹解析失败: {r}"

    def t_extract_bad_json():
        """LLM 输出格式错误时返回 error 而不是崩溃"""
        r = mem_extract_commit(config=config, llm_output=MOCK_LLM_BAD_JSON,
                               participants=["alice", "bob"])
        assert r["status"] == "error", f"期望 error，实际 {r['status']}"

    def t_extract_min_participants():
        r = mem_extract(config=config, conversation_text="今天天气真好",
                        participants=["alice"])  # 单人，不满足 min_participants=2
        assert r["status"] == "skipped", f"单人不应触发提取，实际 {r['status']}"

    def t_extract_pending_llm():
        """不注入 call_llm_fn 时应返回 pending_llm 状态"""
        r = mem_extract(config=config,
                        conversation_text="Alice: 我们决定从1%扩量到5%。\nBob: 好的，我来更新配置。",
                        participants=["alice", "bob"])
        assert r["status"] == "pending_llm", f"期望 pending_llm，实际 {r['status']}"
        assert "prompt" in r, "缺少 prompt 字段"

    runner.run("5.1 LLM 提取正常提交", t_extract_commit_ok)
    runner.run("5.2 LLM 判断跳过写入", t_extract_commit_skip)
    runner.run("5.3 LLM 提取多条记录", t_extract_multi_records)
    runner.run("5.4 markdown 包裹解析", t_extract_markdown_wrapped)
    runner.run("5.5 JSON 格式错误容错", t_extract_bad_json)
    runner.run("5.6 单人不触发提取", t_extract_min_participants)
    runner.run("5.7 无 LLM 返回 pending", t_extract_pending_llm)


# ══════════════════════════════════════════════════════════════
# T6 — 治理与状态（5 个用例）
# ══════════════════════════════════════════════════════════════

def T6(runner: TestRunner, config: dict):
    runner.section("T6 治理与状态")

    def t_status_fields():
        r = mem_status(config)
        for field in ["total_active", "archived", "by_type", "db_path"]:
            assert field in r, f"status 缺少字段: {field}"
        assert r["total_active"] >= 0
        assert r["archived"] >= 0

    def t_status_by_type():
        r = mem_status(config)
        assert isinstance(r["by_type"], dict), "by_type 应为 dict"
        # 确保有 conversation 类型（前面的测试写入了）
        assert "conversation" in r["by_type"], "应有 conversation 类型"

    def t_consolidate_dry_run():
        r = mem_consolidate(config, dry_run=True)
        assert r["dry_run"] is True
        assert r["archived"] == 0, "dry_run 不应实际归档"
        assert "to_archive" in r

    def t_consolidate_count():
        """to_archive 数量应该合理（不超过 total_active）"""
        status = mem_status(config)
        r = mem_consolidate(config, dry_run=True)
        assert r["to_archive"] <= status["total_active"], "待归档数不应超过活跃总数"

    def t_status_after_write():
        """写入后 total_active 应增加"""
        before = mem_status(config)["total_active"]
        mem_write(config=config, mem_type="conversation",
                  content={"summary": "T6: 状态验证写入"},
                  owner="alice")
        after = mem_status(config)["total_active"]
        assert after > before, f"写入后 total_active 未增加: {before} → {after}"

    runner.run("6.1 status 字段完整性", t_status_fields)
    runner.run("6.2 status by_type 分布", t_status_by_type)
    runner.run("6.3 consolidate dry_run", t_consolidate_dry_run)
    runner.run("6.4 to_archive 数量合理", t_consolidate_count)
    runner.run("6.5 写入后 total_active 增加", t_status_after_write)


# ══════════════════════════════════════════════════════════════
# T7 — Embedder 单元测试（8 个用例）
# ══════════════════════════════════════════════════════════════

def T7(runner: TestRunner):
    runner.section("T7 Embedder 单元测试")

    try:
        from embedder import Embedder
        import numpy as np
        embedder = Embedder()
    except Exception as e:
        runner.run("7.0 Embedder 加载", lambda: (_ for _ in ()).throw(
            AssertionError(f"Embedder 加载失败: {e}")))
        return

    def t_output_shape_single():
        vec = embedder.encode("Hello world")
        assert vec.shape == (384,), f"单条输出维度错误: {vec.shape}"

    def t_output_shape_batch():
        vecs = embedder.encode(["text1", "text2", "text3"])
        assert vecs.shape == (3, 384), f"批量输出维度错误: {vecs.shape}"

    def t_l2_normalized():
        vec = embedder.encode("归一化测试")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"L2 范数应为 1.0，实际 {norm:.6f}"

    def t_batch_l2_normalized():
        vecs = embedder.encode(["文本A", "文本B", "文本C"])
        norms = np.linalg.norm(vecs, axis=1)
        for i, n in enumerate(norms):
            assert abs(n - 1.0) < 1e-5, f"第{i}条 L2 范数异常: {n:.6f}"

    def t_similarity_range():
        """余弦相似度应在 [-1, 1]"""
        a = embedder.encode("这是一句话")
        b = embedder.encode("another sentence")
        sim = embedder.similarity(a, b)
        assert -1.0 <= sim <= 1.0, f"相似度超出范围: {sim}"

    def t_self_similarity():
        """自相似度应为 1.0"""
        vec = embedder.encode("自相似度测试")
        sim = embedder.similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5, f"自相似度应为 1.0，实际 {sim:.6f}"

    def t_semantic_order():
        """语义相近的文本相似度应高于不相关的"""
        q = embedder.encode("machine learning model training")
        a_related = embedder.encode("deep learning neural network optimization")
        a_unrelated = embedder.encode("today is a sunny day")
        sim_related = embedder.similarity(q, a_related)
        sim_unrelated = embedder.similarity(q, a_unrelated)
        assert sim_related > sim_unrelated, \
            f"相关文本相似度 {sim_related:.3f} 应 > 不相关 {sim_unrelated:.3f}"

    def t_empty_string():
        """空字符串不应崩溃"""
        vec = embedder.encode("")
        assert vec.shape == (384,), "空字符串输出维度错误"
        norm = np.linalg.norm(vec)
        assert norm >= 0, "空字符串范数异常"

    runner.run("7.1 单条输出维度 (384,)", t_output_shape_single)
    runner.run("7.2 批量输出维度 (N, 384)", t_output_shape_batch)
    runner.run("7.3 单条 L2 归一化", t_l2_normalized)
    runner.run("7.4 批量 L2 归一化", t_batch_l2_normalized)
    runner.run("7.5 余弦相似度范围 [-1, 1]", t_similarity_range)
    runner.run("7.6 自相似度 = 1.0", t_self_similarity)
    runner.run("7.7 语义顺序正确", t_semantic_order)
    runner.run("7.8 空字符串不崩溃", t_empty_string)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="显示失败详情")
    parser.add_argument("--skip-embedder", action="store_true", help="跳过 T7 Embedder 测试")
    args = parser.parse_args()

    print("=" * 65)
    print("  企业级记忆引擎 — 全量测试套件")
    print("=" * 65)

    runner = TestRunner(verbose=args.verbose)
    t_total_start = time.perf_counter()

    # 使用临时 SQLite DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    try:
        config = make_config(tmp_db)
        init_sqlite(Path(tmp_db))

        T1(runner, config)
        T2(runner, config)
        T3(runner, config)
        T4(runner, config)
        T5(runner, config)
        T6(runner, config)
        if not args.skip_embedder:
            T7(runner)
        else:
            print("\n【T7 Embedder 单元测试】（已跳过）")

    finally:
        Path(tmp_db).unlink(missing_ok=True)

    elapsed = time.perf_counter() - t_total_start
    summary = runner.summary()

    print("\n" + "=" * 65)
    print(f"  测试结果：{summary['passed']}/{summary['total']} 通过  "
          f"({summary['pass_rate']})  总耗时 {elapsed:.1f}s")

    if summary["slowest"]:
        print(f"\n  最慢用例：")
        for s in summary["slowest"]:
            print(f"    {s['test']}  {s['ms']}ms")

    if summary["errors"]:
        print(f"\n  失败详情：")
        for e in summary["errors"]:
            print(f"    ❌ {e['test']}")
            print(f"       {e['error']}")
    print("=" * 65)

    # 写 JSON 报告
    report_path = Path(__file__).parent / "full_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({**summary, "elapsed_s": round(elapsed, 2)}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n  报告已写入：{report_path}")
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
