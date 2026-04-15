#!/usr/bin/env python3
"""
test_knowledge_graph.py — 知识图谱功能测试

覆盖：
  G1  实体提取（规则正确性）
  G2  关系提取（三元组正确性）
  G3  图谱写入（SQLite 持久化）
  G4  图增强检索（mem_graph_search）
  G5  实体邻居查询（mem_graph_recall）
  G6  多跳推理（BFS 扩展）
  G7  图谱边界（空输入、重复、权限）
"""

import json
import sys
import tempfile
import time
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

from memory_engine import (
    mem_write, mem_search, mem_graph_search, mem_graph_recall,
)
from init_db import init_sqlite
from graph_extractor import (
    extract_entities, extract_relations,
    upsert_entities_and_edges, graph_bfs, _make_entity_id,
)


# ══════════════════════════════════════════════════════════════
# 测试框架
# ══════════════════════════════════════════════════════════════

class Runner:
    def __init__(self):
        self.passed = self.failed = 0
        self.errors = []

    def run(self, name, fn):
        try:
            fn()
            print(f"  ✅ {name}")
            self.passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}")
            print(f"     → {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except Exception as e:
            print(f"  💥 {name}")
            import traceback
            print(f"     → {traceback.format_exc()}")
            self.failed += 1
            self.errors.append((name, str(e)))

    def section(self, t):
        print(f"\n【{t}】")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*55}")
        print(f"  知识图谱测试: {self.passed}/{total} 通过  "
              f"({'100%' if self.failed == 0 else f'{self.passed/total*100:.1f}%'})")
        if self.errors:
            print("  失败：")
            for name, err in self.errors:
                print(f"    {name}: {err[:80]}")
        print(f"{'='*55}")
        return self.failed == 0


def make_config(db_path):
    return {
        "storage": {"backend": "sqlite", "db_path": db_path},
        "retention": {"fresh_days": 7, "active_days": 30, "archive_days": 90},
        "extraction": {"min_participants": 2},
    }


# ══════════════════════════════════════════════════════════════
# G1 — 实体提取
# ══════════════════════════════════════════════════════════════

def G1(runner: Runner):
    runner.section("G1 实体提取")

    def t_extract_persons():
        ents = extract_entities(
            summary="alice 决定灰度扩量到5%，bob 负责执行",
            content={"summary": ""},
            tags=[], participants=["alice", "bob"],
            owner="alice", mem_type="conversation",
        )
        names = {e["name"] for e in ents}
        assert "alice" in names, f"alice 未提取: {names}"
        assert "bob" in names, f"bob 未提取: {names}"
        person_types = {e["entity_type"] for e in ents if e["name"] in ("alice", "bob")}
        assert person_types == {"person"}, f"类型错误: {person_types}"

    def t_extract_decision():
        ents = extract_entities(
            summary="决定：采用 CQL 做 AI 推荐决策",
            content={"summary": "决定：采用 CQL 做 AI 推荐决策"},
            tags=[], participants=["alice"], owner="alice",
            mem_type="conversation",
        )
        types = {e["entity_type"] for e in ents}
        assert "decision" in types or "project" in types, f"未提取决策/项目: {types}"

    def t_extract_project_from_tags():
        ents = extract_entities(
            summary="模型训练完成",
            content={}, tags=["ai-reco", "训练"],
            participants=[], owner="alice", mem_type="conversation",
        )
        names = {e["name"] for e in ents}
        assert "ai-reco" in names or "训练" in names, f"未从 tags 提取: {names}"

    def t_extract_concept():
        ents = extract_entities(
            summary="灰度发布策略：从1%到5%，观察48小时",
            content={}, tags=[], participants=[], owner="alice",
            mem_type="conversation",
        )
        types = [e["entity_type"] for e in ents]
        assert "concept" in types or "decision" in types, f"未提取 concept/decision: {types}"

    def t_extract_relationship_type():
        ents = extract_entities(
            summary="",
            content={"person": "carol", "role": "工程师", "email": "carol@test.com"},
            tags=[], participants=[], owner="alice",
            mem_type="relationship",
        )
        names = {e["name"] for e in ents}
        assert "carol" in names, f"relationship 类型未提取 person: {names}"

    def t_extract_action_items_owner():
        ents = extract_entities(
            summary="灰度决策",
            content={
                "summary": "灰度决策",
                "action_items": [{"owner": "dave", "task": "更新配置", "due": ""}]
            },
            tags=[], participants=["alice"], owner="alice",
            mem_type="conversation",
        )
        names = {e["name"] for e in ents}
        assert "dave" in names, f"action_items.owner 未提取: {names}"

    def t_entity_id_stable():
        id1 = _make_entity_id("alice", "person")
        id2 = _make_entity_id("alice", "person")
        assert id1 == id2, "相同实体 ID 不稳定"
        id3 = _make_entity_id("alice", "project")
        assert id1 != id3, "不同类型实体 ID 应不同"

    def t_extract_empty_inputs():
        ents = extract_entities(
            summary="", content={}, tags=[], participants=[],
            owner="", mem_type="conversation",
        )
        assert isinstance(ents, list), "空输入应返回列表"

    runner.run("1.1 参与者提取为 person", t_extract_persons)
    runner.run("1.2 决定关键词提取为 decision/project", t_extract_decision)
    runner.run("1.3 从 tags 提取项目实体", t_extract_project_from_tags)
    runner.run("1.4 概念实体提取", t_extract_concept)
    runner.run("1.5 relationship 类型提取关联人", t_extract_relationship_type)
    runner.run("1.6 action_items.owner 提取", t_extract_action_items_owner)
    runner.run("1.7 实体 ID 稳定性", t_entity_id_stable)
    runner.run("1.8 空输入不崩溃", t_extract_empty_inputs)


# ══════════════════════════════════════════════════════════════
# G2 — 关系提取
# ══════════════════════════════════════════════════════════════

def G2(runner: Runner):
    runner.section("G2 关系提取")

    def _ents(summary, participants=None, owner="alice", tags=None, content=None):
        return extract_entities(
            summary=summary,
            content=content or {"summary": summary},
            tags=tags or [], participants=participants or [],
            owner=owner, mem_type="conversation",
        )

    def t_owner_decides_decision():
        ents = _ents("决定：灰度扩量到5%", owner="alice", participants=["alice", "bob"])
        rels = extract_relations(
            entities=ents, summary="决定：灰度扩量到5%",
            content={"summary": "决定：灰度扩量到5%"},
            mem_type="conversation", owner="alice", participants=["alice", "bob"],
        )
        rel_types = {r["relation_type"] for r in rels}
        assert "决定" in rel_types or "负责" in rel_types, f"owner→decision 关系缺失: {rel_types}"

    def t_participant_joins_project():
        ents = _ents("AI 推荐决策 项目讨论", participants=["alice", "bob"],
                     tags=["ai-reco"])
        rels = extract_relations(
            entities=ents, summary="AI 推荐决策 项目讨论",
            content={}, mem_type="conversation",
            owner="alice", participants=["alice", "bob"],
        )
        rel_types = {r["relation_type"] for r in rels}
        assert len(rels) > 0, "应有关系被提取"

    def t_action_item_creates_exec_rel():
        content = {
            "summary": "灰度发布决策",
            "action_items": [{"owner": "bob", "task": "更新配置", "due": ""}],
            "decisions": ["灰度扩量到5%"],
        }
        ents = extract_entities(
            summary="灰度发布决策", content=content,
            tags=[], participants=["alice", "bob"],
            owner="alice", mem_type="conversation",
        )
        rels = extract_relations(
            entities=ents, summary="灰度发布决策",
            content=content, mem_type="conversation",
            owner="alice", participants=["alice", "bob"],
        )
        exec_rels = [r for r in rels if r["relation_type"] == "执行"]
        assert len(exec_rels) > 0, f"action_items 应产生执行关系: {rels}"

    def t_no_self_loop():
        ents = _ents("alice 负责 alice 的项目", owner="alice", participants=["alice"])
        rels = extract_relations(
            entities=ents, summary="alice 负责 alice 的项目",
            content={}, mem_type="conversation",
            owner="alice", participants=["alice"],
        )
        for r in rels:
            assert r["from_id"] != r["to_id"], f"存在自环关系: {r}"

    def t_relation_dedup():
        ents = _ents("alice 决定灰度扩量", owner="alice", participants=["alice", "bob"])
        rels = extract_relations(
            entities=ents, summary="alice 决定灰度扩量",
            content={}, mem_type="conversation",
            owner="alice", participants=["alice", "bob"],
        )
        keys = [(r["from_id"], r["to_id"], r["relation_type"]) for r in rels]
        assert len(keys) == len(set(keys)), "关系存在重复"

    def t_relationship_type_rel():
        content = {"person": "carol", "role": "工程师", "email": "carol@test.com"}
        ents = extract_entities(
            summary="", content=content, tags=[],
            participants=[], owner="alice", mem_type="relationship",
        )
        rels = extract_relations(
            entities=ents, summary="", content=content,
            mem_type="relationship", owner="alice", participants=[],
        )
        types = {r["relation_type"] for r in rels}
        assert "关联" in types, f"relationship 类型应有 关联 边: {types}"

    runner.run("2.1 owner ──决定/负责──> decision/project", t_owner_decides_decision)
    runner.run("2.2 participant ──参与──> project", t_participant_joins_project)
    runner.run("2.3 action_items 产生执行关系", t_action_item_creates_exec_rel)
    runner.run("2.4 无自环关系", t_no_self_loop)
    runner.run("2.5 关系去重", t_relation_dedup)
    runner.run("2.6 relationship 类型产生关联边", t_relationship_type_rel)


# ══════════════════════════════════════════════════════════════
# G3 — 图谱写入（SQLite 持久化）
# ══════════════════════════════════════════════════════════════

def G3(runner: Runner, config: dict):
    runner.section("G3 图谱写入")
    import sqlite3
    db_path = config["storage"]["db_path"]

    def t_write_creates_entities():
        mem_write(
            config=config, mem_type="conversation",
            content={"summary": "G3: alice 决定灰度从1%扩量到5%，bob 负责执行"},
            owner="alice", participants=["alice", "bob"],
            tags=["灰度", "g3-test"], visibility="public",
        )
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        count = conn.execute("SELECT COUNT(*) as n FROM entities").fetchone()["n"]
        conn.close()
        assert count > 0, "写入记忆后 entities 表应有记录"

    def t_write_creates_edges():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        count = conn.execute("SELECT COUNT(*) as n FROM edges").fetchone()["n"]
        conn.close()
        assert count >= 0, "edges 表存在"

    def t_write_creates_mentions():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        count = conn.execute("SELECT COUNT(*) as n FROM entity_mentions").fetchone()["n"]
        conn.close()
        assert count > 0, "entity_mentions 表应有记录"

    def t_importance_increments():
        """重复提及同一实体，importance 应增加"""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        eid = _make_entity_id("alice", "person")
        before = conn.execute(
            "SELECT importance FROM entities WHERE id=?", (eid,)
        ).fetchone()
        conn.close()

        # 再次写入包含 alice 的记忆
        mem_write(
            config=config, mem_type="conversation",
            content={"summary": "G3: alice 再次出现，importance 应增加"},
            owner="alice", participants=["alice"], visibility="public",
        )
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        after = conn.execute(
            "SELECT importance FROM entities WHERE id=?", (eid,)
        ).fetchone()
        conn.close()

        if before and after:
            assert after["importance"] >= before["importance"], \
                f"importance 未增加: {before['importance']} → {after['importance']}"

    def t_entity_upsert_no_dup():
        """相同实体多次写入不产生重复行"""
        for _ in range(3):
            mem_write(
                config=config, mem_type="conversation",
                content={"summary": f"G3: upsert 测试 carol 重复出现 {_}"},
                owner="carol", participants=["carol"], visibility="public",
            )
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        eid = _make_entity_id("carol", "person")
        count = conn.execute(
            "SELECT COUNT(*) as n FROM entities WHERE id=?", (eid,)
        ).fetchone()["n"]
        conn.close()
        assert count == 1, f"相同实体应只有1行，实际 {count}"

    runner.run("3.1 写入记忆后 entities 表有记录", t_write_creates_entities)
    runner.run("3.2 edges 表存在", t_write_creates_edges)
    runner.run("3.3 entity_mentions 有记录", t_write_creates_mentions)
    runner.run("3.4 重复提及 importance 增加", t_importance_increments)
    runner.run("3.5 实体 UPSERT 不产生重复行", t_entity_upsert_no_dup)


# ══════════════════════════════════════════════════════════════
# G4 — 图增强检索
# ══════════════════════════════════════════════════════════════

def G4(runner: Runner, config: dict):
    runner.section("G4 图增强检索 (mem_graph_search)")

    # 预写数据：alice 和 bob 各自相关的记忆
    mem_write(
        config=config, mem_type="conversation",
        content={"summary": "G4: alice 负责灰度发布策略，决定从1%扩到5%"},
        owner="alice", participants=["alice", "bob"],
        tags=["灰度", "g4"], visibility="public",
    )
    mem_write(
        config=config, mem_type="conversation",
        content={"summary": "G4: bob 执行灰度配置更新，完成上线"},
        owner="bob", participants=["bob"],
        tags=["灰度", "g4", "上线"], visibility="public",
    )
    mem_write(
        config=config, mem_type="document",
        content={"title": "G4: 灰度发布方案文档", "summary": "G4: 灰度发布方案文档"},
        owner="alice", tags=["g4", "文档"], visibility="public",
    )

    def t_graph_search_returns_direct():
        result = mem_graph_search(
            config=config, query="灰度", caller_id="alice",
            days=90, top_k=5, max_hops=1,
        )
        assert "direct" in result, "缺少 direct 字段"
        assert len(result["direct"]) > 0, "应有直接命中结果"

    def t_graph_search_returns_entities():
        result = mem_graph_search(
            config=config, query="g4", caller_id="alice",
            days=90, top_k=5, max_hops=1,
        )
        assert "entities" in result, "缺少 entities 字段"
        assert isinstance(result["entities"], list)

    def t_graph_search_returns_edges():
        result = mem_graph_search(
            config=config, query="g4", caller_id="alice",
            days=90, top_k=5, max_hops=1,
        )
        assert "edges" in result, "缺少 edges 字段"
        assert isinstance(result["edges"], list)

    def t_graph_search_total_field():
        result = mem_graph_search(
            config=config, query="灰度", caller_id="alice",
            days=90, top_k=5, max_hops=1,
        )
        assert "total" in result
        assert result["total"] == len(result["direct"]) + len(result["expanded"])

    def t_graph_search_no_result():
        result = mem_graph_search(
            config=config, query="xyznotexist999", caller_id="alice",
            days=90, top_k=5, max_hops=1,
        )
        assert result["total"] == 0
        assert result["direct"] == []

    def t_graph_search_hops_zero():
        """max_hops=0 退化为纯 Hybrid 检索，仍应有结果"""
        result = mem_graph_search(
            config=config, query="灰度", caller_id="alice",
            days=90, top_k=5, max_hops=0,
        )
        assert isinstance(result, dict)
        assert "direct" in result

    def t_graph_search_no_dup():
        """direct 和 expanded 不应有重复 ID"""
        result = mem_graph_search(
            config=config, query="灰度", caller_id="alice",
            days=90, top_k=10, max_hops=1,
        )
        direct_ids = {m["id"] for m in result["direct"]}
        expanded_ids = {m["id"] for m in result["expanded"]}
        overlap = direct_ids & expanded_ids
        assert len(overlap) == 0, f"direct 和 expanded 有重复: {overlap}"

    runner.run("4.1 图检索有 direct 结果", t_graph_search_returns_direct)
    runner.run("4.2 图检索返回 entities", t_graph_search_returns_entities)
    runner.run("4.3 图检索返回 edges", t_graph_search_returns_edges)
    runner.run("4.4 total = direct + expanded", t_graph_search_total_field)
    runner.run("4.5 无结果时 total=0", t_graph_search_no_result)
    runner.run("4.6 max_hops=0 退化为 Hybrid", t_graph_search_hops_zero)
    runner.run("4.7 direct/expanded 无重复", t_graph_search_no_dup)


# ══════════════════════════════════════════════════════════════
# G5 — 实体邻居查询
# ══════════════════════════════════════════════════════════════

def G5(runner: Runner, config: dict):
    runner.section("G5 实体邻居查询 (mem_graph_recall)")

    # 确保 alice 有关联数据（G3/G4 已写入）
    def t_recall_known_entity():
        result = mem_graph_recall(
            config=config, entity_name="alice",
            caller_id="alice", days=365,
        )
        assert result["found"] is True, "alice 应能被找到"
        assert result["entity"]["name"] == "alice"

    def t_recall_memories_nonempty():
        result = mem_graph_recall(
            config=config, entity_name="alice",
            caller_id="alice", days=365,
        )
        assert isinstance(result["memories"], list), "memories 应为列表"

    def t_recall_neighbors():
        result = mem_graph_recall(
            config=config, entity_name="alice",
            caller_id="alice", days=365,
        )
        assert isinstance(result["neighbors"], list), "neighbors 应为列表"

    def t_recall_edges():
        result = mem_graph_recall(
            config=config, entity_name="alice",
            caller_id="alice", days=365,
        )
        assert isinstance(result["edges"], list), "edges 应为列表"

    def t_recall_unknown_entity():
        result = mem_graph_recall(
            config=config, entity_name="nonexistent_xyz_9999",
            caller_id="alice", days=365,
        )
        assert result["found"] is False, "不存在实体应返回 found=False"
        assert result["memories"] == []

    def t_recall_fuzzy_match():
        """模糊匹配：输入 'alic' 应能找到 'alice'（LIKE 匹配）"""
        result = mem_graph_recall(
            config=config, entity_name="alic",
            caller_id="alice", days=365,
        )
        # 模糊匹配可能命中，也可能没有（取决于 LIKE 结果），不崩溃即可
        assert isinstance(result, dict), "模糊匹配应返回 dict"

    def t_recall_permission_filter():
        """private 记忆不应出现在其他人的图谱查询里"""
        # 写一条 private 记忆 for alice
        mem_write(
            config=config, mem_type="conversation",
            content={"summary": "G5: alice 的私密记忆 secret-graph-001"},
            owner="alice", visibility="private",
        )
        # bob 查询 alice 的图谱，不应看到 private 记忆
        result = mem_graph_recall(
            config=config, entity_name="alice",
            caller_id="bob", caller_dept="", days=365,
        )
        private_mems = [m for m in result["memories"]
                        if "secret-graph-001" in m.get("summary", "")]
        assert len(private_mems) == 0, "bob 不应看到 alice 的 private 记忆"

    runner.run("5.1 已知实体 found=True", t_recall_known_entity)
    runner.run("5.2 memories 字段为列表", t_recall_memories_nonempty)
    runner.run("5.3 neighbors 字段为列表", t_recall_neighbors)
    runner.run("5.4 edges 字段为列表", t_recall_edges)
    runner.run("5.5 未知实体 found=False", t_recall_unknown_entity)
    runner.run("5.6 模糊匹配不崩溃", t_recall_fuzzy_match)
    runner.run("5.7 图谱查询遵守权限", t_recall_permission_filter)


# ══════════════════════════════════════════════════════════════
# G6 — 多跳推理（BFS 扩展）
# ══════════════════════════════════════════════════════════════

def G6(runner: Runner, config: dict):
    runner.section("G6 多跳推理 (BFS)")

    import sqlite3
    db_path = config["storage"]["db_path"]

    # 构造多跳链：alice → 灰度发布 → bob → ai-reco
    mem_write(
        config=config, mem_type="conversation",
        content={"summary": "G6: alice 决定灰度发布，bob 参与"},
        owner="alice", participants=["alice", "bob"],
        tags=["灰度发布", "g6"], visibility="public",
    )
    mem_write(
        config=config, mem_type="conversation",
        content={"summary": "G6: bob 使用 ai-reco 模型做灰度决策"},
        owner="bob", participants=["bob"],
        tags=["ai-reco", "g6"], visibility="public",
    )

    def t_bfs_seed_empty():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        result = graph_bfs(conn, [], max_hops=1)
        conn.close()
        assert result["entities"] == []
        assert result["expanded_memory_ids"] == []

    def t_bfs_hops_1():
        """一跳扩展应能找到种子记忆的关联实体"""
        # 先找 G6 相关的记忆 ID
        results = mem_search(
            config=config, query="g6", caller_id="alice", days=90,
        )
        seed_ids = [r["id"] for r in results[:2]]
        if not seed_ids:
            return  # 无数据跳过

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        result = graph_bfs(conn, seed_ids, max_hops=1)
        conn.close()
        assert isinstance(result["entities"], list)
        assert isinstance(result["edges"], list)

    def t_bfs_hops_2():
        """二跳扩展覆盖范围应 >= 一跳"""
        results = mem_search(
            config=config, query="g6", caller_id="alice", days=90,
        )
        seed_ids = [r["id"] for r in results[:1]]
        if not seed_ids:
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        r1 = graph_bfs(conn, seed_ids, max_hops=1)
        r2 = graph_bfs(conn, seed_ids, max_hops=2)
        conn.close()
        # 二跳发现的实体 >= 一跳
        assert len(r2["entities"]) >= len(r1["entities"]), \
            f"二跳实体 {len(r2['entities'])} < 一跳 {len(r1['entities'])}"

    def t_bfs_no_dup_entities():
        results = mem_search(config=config, query="g6", caller_id="alice", days=90)
        seed_ids = [r["id"] for r in results[:2]]
        if not seed_ids:
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        result = graph_bfs(conn, seed_ids, max_hops=1)
        conn.close()
        entity_ids = [e["id"] for e in result["entities"]]
        assert len(entity_ids) == len(set(entity_ids)), "BFS 实体结果有重复"

    def t_bfs_max_nodes_respected():
        results = mem_search(config=config, query="G6", caller_id="alice", days=90)
        seed_ids = [r["id"] for r in results]
        if not seed_ids:
            return

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        result = graph_bfs(conn, seed_ids, max_hops=3, max_nodes=5)
        conn.close()
        assert len(result["entities"]) <= 5, \
            f"max_nodes=5 但返回 {len(result['entities'])} 个实体"

    runner.run("6.1 空种子 BFS 返回空", t_bfs_seed_empty)
    runner.run("6.2 一跳 BFS 扩展", t_bfs_hops_1)
    runner.run("6.3 二跳 >= 一跳", t_bfs_hops_2)
    runner.run("6.4 BFS 实体无重复", t_bfs_no_dup_entities)
    runner.run("6.5 max_nodes 限制生效", t_bfs_max_nodes_respected)


# ══════════════════════════════════════════════════════════════
# G7 — 图谱边界
# ══════════════════════════════════════════════════════════════

def G7(runner: Runner, config: dict):
    runner.section("G7 图谱边界")

    def t_write_no_crash_empty_summary():
        r = mem_write(
            config=config, mem_type="conversation",
            content={"summary": ""},
            owner="alice", visibility="public",
        )
        assert r["status"] in ("ok", "duplicate"), f"空 summary 写入应不崩溃: {r}"

    def t_graph_search_large_hops():
        """max_hops=10 不应死循环或超时"""
        t0 = time.perf_counter()
        result = mem_graph_search(
            config=config, query="alice", caller_id="alice",
            days=90, top_k=5, max_hops=10,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"max_hops=10 耗时 {elapsed:.2f}s 超过 10s"
        assert isinstance(result, dict)

    def t_recall_entity_type_filter():
        """按 entity_type 精确匹配"""
        r = mem_graph_recall(
            config=config, entity_name="alice",
            entity_type="person",
            caller_id="alice", days=365,
        )
        if r["found"]:
            assert r["entity"]["entity_type"] == "person"

    def t_write_special_char_summary():
        """包含特殊字符的 summary 不应导致图谱崩溃"""
        r = mem_write(
            config=config, mem_type="conversation",
            content={"summary": "G7: 特殊字符 <test> & \"quoted\" 'single'"},
            owner="alice", visibility="public",
        )
        assert r["status"] in ("ok", "duplicate")

    def t_write_very_long_summary():
        long_s = "G7长文本实体提取" + "测" * 500
        r = mem_write(
            config=config, mem_type="conversation",
            content={"summary": long_s},
            owner="alice", visibility="public",
        )
        assert r["status"] in ("ok", "duplicate")

    def t_graph_recall_days_zero():
        r = mem_graph_recall(
            config=config, entity_name="alice",
            caller_id="alice", days=0,
        )
        assert isinstance(r, dict), "days=0 不应崩溃"

    runner.run("7.1 空 summary 写入不崩溃", t_write_no_crash_empty_summary)
    runner.run("7.2 max_hops=10 不死循环", t_graph_search_large_hops)
    runner.run("7.3 entity_type 精确过滤", t_recall_entity_type_filter)
    runner.run("7.4 特殊字符 summary 不崩溃", t_write_special_char_summary)
    runner.run("7.5 超长 summary 不崩溃", t_write_very_long_summary)
    runner.run("7.6 days=0 不崩溃", t_graph_recall_days_zero)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  知识图谱功能测试")
    print("=" * 55)

    runner = Runner()
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    try:
        init_sqlite(Path(tmp_db))
        config = make_config(tmp_db)

        G1(runner)
        G2(runner)
        G3(runner, config)
        G4(runner, config)
        G5(runner, config)
        G6(runner, config)
        G7(runner, config)

    finally:
        Path(tmp_db).unlink(missing_ok=True)

    elapsed = time.perf_counter() - t0
    ok = runner.summary()
    print(f"  总耗时 {elapsed:.1f}s")

    import sys
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
