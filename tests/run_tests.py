#!/usr/bin/env python3
"""
run_tests.py — 企业记忆引擎评测用例
覆盖：写入、去重、检索、权限过滤、自动提取、治理六大场景

运行：python3 tests/run_tests.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 将 scripts/ 加入 path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from memory_engine import (
    mem_write, mem_search, mem_recall,
    mem_extract_commit, mem_status, mem_consolidate,
    load_config, get_db_path,
)
from init_db import init_sqlite


# ══════════════════════════════════════════════════════════════
# 测试基础设施
# ══════════════════════════════════════════════════════════════

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name: str, fn):
        try:
            fn()
            print(f"  ✅ {name}")
            self.passed += 1
        except AssertionError as e:
            print(f"  ❌ {name} — {e}")
            self.failed += 1
            self.errors.append({"test": name, "error": str(e)})
        except Exception as e:
            print(f"  💥 {name} — 异常：{e}")
            self.failed += 1
            self.errors.append({"test": name, "error": f"Exception: {e}"})

    def summary(self) -> dict:
        total = self.passed + self.failed
        return {
            "total": total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.passed/total*100:.1f}%" if total else "0%",
            "errors": self.errors,
        }


def make_test_config(db_path: str) -> dict:
    """生成测试用配置（指向临时 DB）"""
    return {
        "storage": {"backend": "sqlite", "db_path": db_path},
        "retention": {"fresh_days": 7, "active_days": 30, "archive_days": 90},
        "extraction": {"min_participants": 2},
        "multi_user": {"enabled": True, "default_visibility": "team"},
        "search": {"hybrid_alpha": 0.6, "top_k": 10, "time_decay": True},
    }


# ══════════════════════════════════════════════════════════════
# T1 — 基础写入测试
# ══════════════════════════════════════════════════════════════

def test_write_conversation(config):
    result = mem_write(
        config=config,
        mem_type="conversation",
        content={
            "summary": "确认 AI 推荐决策 灰度比例为 1%",
            "decisions": ["灰度从 1% 开始，48h 后评估扩量"],
            "action_items": [{"owner": "user_a", "task": "部署 20260414 版本", "due": "2026-04-15"}],
        },
        owner="user_a",
        participants=["user_a", "user_b"],
        tags=["ai-reco", "灰度"],
        department="算法工程",
    )
    assert result["status"] == "ok", f"写入失败：{result}"
    assert result["id"].startswith("conv_"), f"ID 格式错误：{result['id']}"
    return result["id"]


def test_write_document(config):
    result = mem_write(
        config=config,
        mem_type="document",
        content={
            "title": "强化学习推荐决策方案评估报告",
            "summary": "20260309 版本：Bitrate-4.3%，丢帧-10.65%",
            "url": "https://docs.techcorp.com/doc/0a526f...",
            "key_facts": ["Bitrate下降 4.3%", "1080p 覆盖率提升 5.23%"],
        },
        owner="user_a",
        tags=["评估报告", "20260309"],
        department="算法工程",
    )
    assert result["status"] == "ok", f"文档写入失败：{result}"
    return result["id"]


def test_write_relationship(config):
    result = mem_write(
        config=config,
        mem_type="relationship",
        content={
            "person": "user_b",
            "email": "user_b@company.com",
            "department": "产品",
            "role": "产品经理",
            "expertise": ["需求分析", "数据驱动"],
        },
        owner="user_a",
        tags=["产品侧", "合作方"],
        department="算法工程",
        visibility="public",
    )
    assert result["status"] == "ok", f"关系写入失败：{result}"
    return result["id"]


def test_write_event(config):
    result = mem_write(
        config=config,
        mem_type="event",
        content={
            "title": "AI 推荐决策 灰度启动评审会",
            "time": "2026-04-14T14:00:00+08:00",
            "outcome": "批准灰度，比例 1%，观察 48h",
        },
        owner="user_a",
        participants=["user_a", "user_b", "user_c"],
        tags=["灰度", "里程碑"],
        department="算法工程",
        visibility="public",
    )
    assert result["status"] == "ok", f"事件写入失败：{result}"
    return result["id"]


# ══════════════════════════════════════════════════════════════
# T2 — 去重测试
# ══════════════════════════════════════════════════════════════

def test_dedup_exact(config):
    """完全相同内容写两次，第二次应被去重"""
    # 每次传入独立的 dict，确保 mem_write 内的 setdefault 修改不影响哈希一致性
    # 去重基于 content_hash（排除 created_at），所以相同业务内容应被识别为重复
    base = {"summary": "去重测试用例 - 这条记忆不应重复写入"}
    mem_write(config=config, mem_type="conversation", content=dict(base),
              owner="alice", department="测试部")
    result2 = mem_write(config=config, mem_type="conversation", content=dict(base),
                        owner="alice", department="测试部")
    assert result2["status"] == "duplicate", f"去重失败，应为 duplicate，实际：{result2['status']}"


# ══════════════════════════════════════════════════════════════
# T3 — 检索测试
# ══════════════════════════════════════════════════════════════

def test_search_keyword(config):
    """关键词检索：能找到包含 '灰度' 的记忆"""
    # 先写入一条
    mem_write(
        config=config,
        mem_type="conversation",
        content={"summary": "灰度发布检索验证用例 keyword-test-001"},
        owner="searcher",
        tags=["灰度", "检索测试"],
        department="算法工程",
        visibility="public",
    )
    
    results = mem_search(
        config=config,
        query="灰度",
        caller_id="searcher",
        caller_dept="算法工程",
        days=90,
        top_k=10,
    )
    assert len(results) > 0, "关键词检索无结果"
    summaries = [r.get("summary", "") for r in results]
    assert any("灰度" in s for s in summaries), f"检索结果不包含预期关键词，结果：{summaries}"


def test_search_no_result(config):
    """检索一个不存在的内容应返回空"""
    results = mem_search(
        config=config,
        query="这段内容绝对不存在xyzabc999",
        caller_id="alice",
        days=90,
    )
    assert len(results) == 0, f"检索不存在内容应返回空，实际返回 {len(results)} 条"


def test_recall_entity(config):
    """Recall：能找到某人的所有记忆"""
    # 写入一条 user_a 相关
    mem_write(
        config=config,
        mem_type="conversation",
        content={"summary": "recall 测试 user_a 参与的会议"},
        owner="user_a",
        participants=["user_a", "recall_partner"],
        tags=["recall-test"],
        department="算法工程",
        visibility="public",
    )
    
    result = mem_recall(
        config=config,
        entity="user_a",
        depth=20,
        caller_dept="算法工程",
    )
    assert result["total"] > 0, "recall user_a 应有记录"
    assert result["entity"] == "user_a"


# ══════════════════════════════════════════════════════════════
# T4 — 权限过滤测试
# ══════════════════════════════════════════════════════════════

def test_permission_private(config):
    """private 记忆不应被非 owner 检索到"""
    # alice 写一条 private 记忆
    mem_write(
        config=config,
        mem_type="conversation",
        content={"summary": "私密记忆 permission-private-test-001 alice-only"},
        owner="alice",
        tags=["私密", "权限测试"],
        visibility="private",
        department="算法工程",
    )
    
    # bob 检索，不应看到
    results = mem_search(
        config=config,
        query="私密记忆 permission-private-test-001",
        caller_id="bob",
        caller_dept="算法工程",
        days=90,
    )
    private_found = any("permission-private-test-001" in r.get("summary", "") for r in results)
    assert not private_found, "private 记忆被非 owner 检索到，权限过滤失效"


def test_permission_public_visible(config):
    """public 记忆应被任何人检索到"""
    mem_write(
        config=config,
        mem_type="event",
        content={"summary": "公开事件 permission-public-test-001"},
        owner="alice",
        tags=["公开", "权限测试"],
        visibility="public",
        department="算法工程",
    )
    
    # carol（不同部门）也能看到
    results = mem_search(
        config=config,
        query="公开事件 permission-public-test-001",
        caller_id="carol",
        caller_dept="市场部",
        days=90,
    )
    public_found = any("permission-public-test-001" in r.get("summary", "") for r in results)
    assert public_found, "public 记忆应对所有人可见，但未检索到"


# ══════════════════════════════════════════════════════════════
# T5 — 自动提取测试
# ══════════════════════════════════════════════════════════════

MOCK_LLM_OUTPUT = json.dumps({
    "should_store": True,
    "reason": "包含明确决策和 Action Item",
    "records": [{
        "type": "conversation",
        "summary": "决定将灰度比例从 1% 提升到 5%（自动提取测试）",
        "decisions": ["灰度从 1% → 5%，生效时间 2026-04-15"],
        "action_items": [{"owner": "user_a", "task": "更新灰度配置", "due": "2026-04-15"}],
        "tags": ["灰度", "自动提取"],
        "visibility": "team"
    }]
}, ensure_ascii=False)

MOCK_LLM_OUTPUT_SKIP = json.dumps({
    "should_store": False,
    "reason": "只是日常问候，无实质内容",
    "records": []
}, ensure_ascii=False)


def test_extract_commit_ok(config):
    """正常提取场景：LLM 判断值得存储"""
    result = mem_extract_commit(
        config=config,
        llm_output=MOCK_LLM_OUTPUT,
        participants=["user_a", "user_b"],
        owner="user_a",
        department="算法工程",
    )
    assert result["status"] == "ok", f"提取应成功，实际：{result}"
    assert result["extracted"] == 1, f"应提取 1 条，实际：{result['extracted']}"


def test_extract_commit_skip(config):
    """无效对话：LLM 判断不值得存储"""
    result = mem_extract_commit(
        config=config,
        llm_output=MOCK_LLM_OUTPUT_SKIP,
        participants=["user_a", "user_b"],
    )
    assert result["status"] == "skipped", f"应被跳过，实际：{result['status']}"


def test_extract_min_participants(config):
    """单人对话不触发自动提取"""
    from memory_engine import mem_extract
    result = mem_extract(
        config=config,
        conversation_text="我今天很忙",
        participants=["user_a"],  # 只有 1 人
    )
    assert result["status"] == "skipped", f"单人对话应 skipped，实际：{result['status']}"


# ══════════════════════════════════════════════════════════════
# T6 — 状态与治理测试
# ══════════════════════════════════════════════════════════════

def test_status(config):
    """status 应返回合法的统计数据"""
    result = mem_status(config)
    assert "total_active" in result, "缺少 total_active 字段"
    assert "by_type" in result, "缺少 by_type 字段"
    assert result["total_active"] >= 0, "total_active 不应为负"


def test_consolidate_dry_run(config):
    """dry_run 不应实际归档"""
    result = mem_consolidate(config, dry_run=True)
    assert result["dry_run"] is True
    assert result["archived"] == 0, "dry_run 不应有实际归档"


# ══════════════════════════════════════════════════════════════
# 主测试流程
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("企业级长程协作记忆引擎 — 评测用例")
    print("=" * 60)
    
    # 使用临时数据库，不污染生产数据
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    
    try:
        config = make_test_config(tmp_db)
        init_sqlite(Path(tmp_db))
        
        runner = TestRunner()
        
        # T1 基础写入
        print("\n【T1】基础写入")
        runner.run("写入对话记忆", lambda: test_write_conversation(config))
        runner.run("写入文档记忆", lambda: test_write_document(config))
        runner.run("写入人际关系", lambda: test_write_relationship(config))
        runner.run("写入事件记忆", lambda: test_write_event(config))
        
        # T2 去重
        print("\n【T2】去重机制")
        runner.run("完全相同内容去重", lambda: test_dedup_exact(config))
        
        # T3 检索
        print("\n【T3】检索能力")
        runner.run("关键词检索命中", lambda: test_search_keyword(config))
        runner.run("检索不存在内容返回空", lambda: test_search_no_result(config))
        runner.run("Recall 实体记忆链", lambda: test_recall_entity(config))
        
        # T4 权限
        print("\n【T4】权限过滤")
        runner.run("private 记忆非 owner 不可见", lambda: test_permission_private(config))
        runner.run("public 记忆跨部门可见", lambda: test_permission_public_visible(config))
        
        # T5 自动提取
        print("\n【T5】自动提取")
        runner.run("LLM 提取正常提交", lambda: test_extract_commit_ok(config))
        runner.run("LLM 判断跳过写入", lambda: test_extract_commit_skip(config))
        runner.run("单人对话不触发提取", lambda: test_extract_min_participants(config))
        
        # T6 治理
        print("\n【T6】状态与治理")
        runner.run("记忆库状态统计", lambda: test_status(config))
        runner.run("干跑归档不实际执行", lambda: test_consolidate_dry_run(config))
        
        # 汇总
        summary = runner.summary()
        print("\n" + "=" * 60)
        print(f"评测结果：{summary['passed']}/{summary['total']} 通过  ({summary['pass_rate']})")
        if summary["errors"]:
            print("\n失败详情：")
            for err in summary["errors"]:
                print(f"  - {err['test']}: {err['error']}")
        print("=" * 60)
        
        # 输出机器可读 JSON 报告
        report_path = Path(__file__).parent / "test_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n评测报告已写入：{report_path}")
        
        sys.exit(0 if summary["failed"] == 0 else 1)
    
    finally:
        # 清理临时数据库
        Path(tmp_db).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
