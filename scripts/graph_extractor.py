#!/usr/bin/env python3
"""
graph_extractor.py — 规则式实体与关系提取器

从记忆的 summary / content / tags / participants 中自动提取：
  - 实体节点：person / project / decision / concept / org
  - 关系三元组：(from_entity, relation_type, to_entity)

设计原则：
  - 零 LLM 依赖（纯规则 + 正则），毫秒级，无网络调用
  - 可选：若传入 call_llm_fn，用 LLM 补充提取（降级安全）
  - 确定性输出，相同输入永远产生相同图结构

关系类型字典：
  执行    person ──执行──> decision/project
  负责    person ──负责──> project/concept
  参与    person ──参与──> event/project
  决定    person ──决定──> decision
  属于    person/project ──属于──> org
  依赖    project ──依赖──> project/concept
  关联    * ──关联──> *  (兜底关系)
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

CST = timezone(timedelta(hours=8))

# ══════════════════════════════════════════════════════════════
# 实体类型识别规则
# ══════════════════════════════════════════════════════════════

# 决策关键词 → decision 实体
_DECISION_PATTERNS = [
    r'决定[：:]\s*(.{4,40})',
    r'确认[：:]\s*(.{4,40})',
    r'约定[：:]\s*(.{4,40})',
    r'方案[是为：:]\s*(.{4,40})',
    r'采用\s*(.{4,30})',
    r'使用\s*(.{4,30})(?:做|来|进行)',
    r'从(\d+%)\s*(?:扩量|灰度|上线)到\s*(\d+%)',
    r'灰度\s*(\d+%)',
]

# 项目/系统关键词 → project 实体
_PROJECT_PATTERNS = [
    r'(offline[\s_-]?rl)',
    r'([\w\-]+推荐决策)',
    r'([\w\-]+模型)',
    r'([\w\-]+系统)',
    r'([\w\-]+项目)',
    r'([\w\-]+引擎)',
    r'([\w\-]+平台)',
    r'(DQN|PPO|IQL|SAC|CQL)',      # RL算法实体
    r'(Wide\s*&?\s*Deep)',
    r'(MiniLM|BERT|GPT|LLM)',
    r'(SQLite|ChromaDB|numpy|ONNX)',
]

# 技术概念 → concept 实体
_CONCEPT_PATTERNS = [
    r'(灰度发布)',
    r'(A/B测试)',
    r'(强化学习)',
    r'(知识蒸馏)',
    r'(知识图谱)',
    r'(向量检索)',
    r'(混合检索|Hybrid)',
    r'(上下文扩展)',
    r'(奖励函数)',
    r'(状态空间)',
    r'(动作空间)',
    r'(Bitrate推荐)',
    r'(卡顿率)',
    r'(丢帧率)',
    r'(QualityScore)',
    r'(recall|precision|MRR)',
]

# 组织/部门 → org 实体
_ORG_PATTERNS = [
    r'(算法工程(?:部门|组|团队)?)',
    r'(基础设施(?:部门|组|团队)?)',
    r'(产品(?:部门|组|团队)?)',
    r'(科研(?:部门|组|团队)?)',
    r'(TechCorp)',
    r'(XHS)',
]

# 动作词 → 关系类型映射
_ACTION_TO_RELATION = [
    (r'负责|主导|owner', '负责'),
    (r'决定|确认|批准|选定', '决定'),
    (r'执行|完成|实现|开发|上线', '执行'),
    (r'参与|出席|与会', '参与'),
    (r'提出|建议|设计', '提出'),
    (r'依赖|使用|基于|调用', '依赖'),
    (r'属于|隶属|在', '属于'),
]


def _now_str() -> str:
    return datetime.now(CST).isoformat()


def _make_entity_id(name: str, entity_type: str) -> str:
    """稳定 ID：基于 name + type 的 hash，相同实体总产生同一个 ID"""
    import hashlib
    slug = hashlib.md5(f"{entity_type}:{name.lower().strip()}".encode()).hexdigest()[:8]
    prefix = {"person": "per", "project": "prj", "decision": "dec",
               "concept": "cpt", "org": "org"}.get(entity_type, "ent")
    return f"ent_{prefix}_{slug}"


def _normalize_name(name: str) -> str:
    """规范化实体名：去多余空白、统一大写缩写"""
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.strip('：:。，,')
    return name


# ══════════════════════════════════════════════════════════════
# 实体提取
# ══════════════════════════════════════════════════════════════

def extract_entities(
    summary: str,
    content: dict,
    tags: list,
    participants: list,
    owner: str,
    mem_type: str,
    department: str = "",
) -> List[Dict]:
    """
    从记忆字段中提取实体列表。

    返回：
    [
      {"id": "ent_per_abc123", "name": "alice", "entity_type": "person",
       "department": "算法工程", "description": "", "role_in_memory": "owner"},
      ...
    ]
    """
    entities: Dict[str, Dict] = {}  # id → entity dict

    def _add(name: str, etype: str, dept: str = "", role: str = "关联",
             description: str = ""):
        name = _normalize_name(name)
        if not name or len(name) < 2:
            return
        eid = _make_entity_id(name, etype)
        if eid not in entities:
            entities[eid] = {
                "id": eid,
                "name": name,
                "entity_type": etype,
                "department": dept or department,
                "description": description,
                "role_in_memory": role,
                "importance": 1.0,
            }

    # 1. 参与者 → person 实体
    for p in (participants or []):
        if p and len(p) >= 2:
            _add(p, "person", department, "参与者")

    # 2. owner → person 实体
    if owner and len(owner) >= 2:
        _add(owner, "person", department, "owner")

    # 3. 标签 → 推断类型
    for tag in (tags or []):
        tag = tag.strip()
        if not tag:
            continue
        # 判断 tag 类型
        if any(re.search(p, tag, re.I) for p in [r'rl$', r'model', r'算法', r'系统', r'引擎']):
            _add(tag, "project", department, "标签")
        elif any(re.search(p, tag, re.I) for p in [r'部门', r'团队', r'组$']):
            _add(tag, "org", department, "标签")
        else:
            _add(tag, "concept", "", "标签")

    # 4. summary 文本模式匹配
    text = summary + " " + json.dumps(content, ensure_ascii=False)

    for pattern in _DECISION_PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            groups = [g for g in m.groups() if g]
            decision_name = "→".join(groups) if groups else m.group(0)[:40]
            _add(decision_name, "decision", department, "决策", summary[:80])

    for pattern in _PROJECT_PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            _add(m.group(1), "project", department, "项目")

    for pattern in _CONCEPT_PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            _add(m.group(1), "concept", "", "概念")

    for pattern in _ORG_PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            _add(m.group(1), "org", "", "组织")

    # 5. action_items 里的 owner → person
    for item in content.get("action_items", []):
        if isinstance(item, dict) and item.get("owner"):
            _add(item["owner"], "person", department, "执行者")

    # 5b. decisions 列表 → decision 实体（直接提取，不走正则）
    for dec in content.get("decisions", []):
        if isinstance(dec, str) and len(dec) >= 2:
            _add(dec[:60], "decision", department, "决策", summary[:80])

    # 6. event 的参与者已通过 participants 覆盖
    # relationship 的 person 字段
    if mem_type == "relationship" and content.get("person"):
        _add(content["person"], "person",
             content.get("department", department), "关联人")

    return list(entities.values())


# ══════════════════════════════════════════════════════════════
# 关系提取
# ══════════════════════════════════════════════════════════════

def extract_relations(
    entities: List[Dict],
    summary: str,
    content: dict,
    mem_type: str,
    owner: str,
    participants: list,
) -> List[Dict]:
    """
    基于实体列表 + 文本，提取关系三元组。

    返回：
    [
      {"from_name": "alice", "from_type": "person",
       "to_name": "ai-reco推荐决策", "to_type": "project",
       "relation_type": "负责"},
      ...
    ]
    """
    relations = []

    # 建名称到实体的快速查找
    by_name: Dict[str, Dict] = {e["name"].lower(): e for e in entities}
    persons = [e for e in entities if e["entity_type"] == "person"]
    projects = [e for e in entities if e["entity_type"] == "project"]
    decisions = [e for e in entities if e["entity_type"] == "decision"]
    concepts = [e for e in entities if e["entity_type"] == "concept"]

    def _add_rel(fe: Dict, te: Dict, rtype: str):
        if fe["id"] != te["id"]:
            relations.append({
                "from_id": fe["id"],
                "from_name": fe["name"],
                "from_type": fe["entity_type"],
                "to_id": te["id"],
                "to_name": te["name"],
                "to_type": te["entity_type"],
                "relation_type": rtype,
            })

    # ── 规则一：owner ──执行/提出──> decision / project ──────────
    owner_ent = None
    for e in persons:
        if e["name"].lower() == owner.lower():
            owner_ent = e
            break
    if owner_ent is None and owner:
        # owner 可能不在 persons 里（未被单独识别）
        owner_ent = {"id": _make_entity_id(owner, "person"),
                     "name": owner, "entity_type": "person"}

    if owner_ent:
        for d in decisions:
            _add_rel(owner_ent, d, "决定")
        for p in projects:
            _add_rel(owner_ent, p, "负责")

    # ── 规则二：participants ──参与──> decision / project ─────────
    for person in persons:
        if person["role_in_memory"] == "参与者":
            for d in decisions:
                _add_rel(person, d, "参与")
            for p in projects:
                _add_rel(person, p, "参与")

    # ── 规则三：执行者(action_items.owner) ──执行──> decision ─────
    for item in content.get("action_items", []):
        if not isinstance(item, dict):
            continue
        item_owner = item.get("owner", "")
        task = item.get("task", "")
        if item_owner:
            # 在已有实体里查找，找不到则创建临时实体对象参与关系提取
            actor = by_name.get(item_owner.lower()) or {
                "id": _make_entity_id(item_owner, "person"),
                "name": item_owner,
                "entity_type": "person",
            }
            # 如果实体不在 by_name 里，加进去以便后续规则命中
            if item_owner.lower() not in by_name:
                by_name[item_owner.lower()] = actor
            for d in decisions:
                _add_rel(actor, d, "执行")
            # 任务描述里提到的 project
            for prj in projects:
                if prj["name"].lower() in task.lower():
                    _add_rel(actor, prj, "执行")

    # ── 规则四：文本动词模式 ────────────────────────────────────
    text = summary
    for action_re, rtype in _ACTION_TO_RELATION:
        # 找到 "person + 动词 + target" 模式
        pattern = rf'(\S{{2,10}})\s*(?:{action_re})\s*(\S{{2,20}})'
        for m in re.finditer(pattern, text):
            subj_name = m.group(1).strip()
            obj_name  = m.group(2).strip()
            subj = by_name.get(subj_name.lower())
            obj  = by_name.get(obj_name.lower())
            if subj and obj:
                _add_rel(subj, obj, rtype)

    # ── 规则五：concept ──依赖──> concept（技术依赖链）─────────
    # 如果 summary 里有 "基于X的Y"、"X结合Y" 等模式
    dep_patterns = [
        r'基于\s*(\S{2,15})\s*的\s*(\S{2,15})',
        r'(\S{2,15})\s*结合\s*(\S{2,15})',
        r'(\S{2,15})\s*替换\s*(\S{2,15})',
        r'(\S{2,15})\s*蒸馏\s*(?:为|到)\s*(\S{2,15})',
    ]
    for dp in dep_patterns:
        for m in re.finditer(dp, text):
            src = by_name.get(m.group(1).lower())
            dst = by_name.get(m.group(2).lower())
            if src and dst:
                _add_rel(src, dst, "依赖")

    # ── 规则六：relationship 类型记忆专属 ────────────────────────
    if mem_type == "relationship":
        person_name = content.get("person", "")
        if person_name and owner:
            ent_a = {"id": _make_entity_id(owner, "person"),
                     "name": owner, "entity_type": "person"}
            ent_b = {"id": _make_entity_id(person_name, "person"),
                     "name": person_name, "entity_type": "person"}
            _add_rel(ent_a, ent_b, "关联")

    # 去重（同 from/to/relation）
    seen = set()
    unique = []
    for r in relations:
        key = (r["from_id"], r["to_id"], r["relation_type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# ══════════════════════════════════════════════════════════════
# 写入图谱（SQLite）
# ══════════════════════════════════════════════════════════════

def upsert_entities_and_edges(
    conn,
    memory_id: str,
    entities: List[Dict],
    relations: List[Dict],
) -> Dict:
    """
    将提取的实体和关系写入知识图谱表。
    实体按 id 去重（upsert）；边按 (from, to, relation, memory) 去重。
    """
    now = _now_str()
    ent_written = 0
    edge_written = 0
    mention_written = 0

    # 1. 写实体（UPSERT：已存在则更新 description/updated_at，importance +0.1）
    for e in entities:
        existing = conn.execute(
            "SELECT id, importance FROM entities WHERE id=?", (e["id"],)
        ).fetchone()

        if existing:
            # 实体已存在：importance 强化（被再次提及）
            conn.execute(
                """UPDATE entities SET
                     importance = MIN(importance + 0.1, 10.0),
                     updated_at = ?
                   WHERE id = ?""",
                (now, e["id"])
            )
        else:
            conn.execute(
                """INSERT INTO entities
                     (id, name, aliases, entity_type, department,
                      description, importance, created_at, updated_at)
                   VALUES (?, ?, '[]', ?, ?, ?, 1.0, ?, ?)""",
                (e["id"], e["name"], e["entity_type"],
                 e.get("department", ""), e.get("description", ""),
                 now, now)
            )
            ent_written += 1

    # 2. 写 entity_mentions
    for e in entities:
        mid = f"men_{memory_id[-6:]}_{e['id'][-6:]}"
        try:
            conn.execute(
                """INSERT OR IGNORE INTO entity_mentions
                     (id, memory_id, entity_id, role, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (mid, memory_id, e["id"], e.get("role_in_memory", "关联"), now)
            )
            mention_written += 1
        except Exception:
            pass

    # 3. 写关系边
    for r in relations:
        edge_id = f"edg_{r['from_id'][-6:]}_{r['to_id'][-6:]}_{memory_id[-6:]}"
        try:
            conn.execute(
                """INSERT OR IGNORE INTO edges
                     (id, from_entity, to_entity, relation_type,
                      memory_id, weight, created_at)
                   VALUES (?, ?, ?, ?, ?, 1.0, ?)""",
                (edge_id, r["from_id"], r["to_id"],
                 r["relation_type"], memory_id, now)
            )
            edge_written += 1
        except Exception:
            pass

    conn.commit()

    return {
        "entities_new": ent_written,
        "mentions": mention_written,
        "edges": edge_written,
    }


# ══════════════════════════════════════════════════════════════
# 图检索：BFS 多跳扩展
# ══════════════════════════════════════════════════════════════

def graph_bfs(
    conn,
    seed_memory_ids: List[str],
    max_hops: int = 1,
    max_nodes: int = 50,
) -> Dict:
    """
    从种子记忆出发，BFS 扩展关联实体和记忆。

    返回：
    {
      "entities": [...],       # 所有发现的实体节点
      "edges": [...],          # 所有边
      "expanded_memory_ids": [...],  # 通过图扩展发现的额外记忆 ID
    }
    """
    if not seed_memory_ids:
        return {"entities": [], "edges": [], "expanded_memory_ids": []}

    # Step 1: 找种子记忆关联的实体
    placeholders = ",".join("?" * len(seed_memory_ids))
    seed_entities = conn.execute(
        f"""SELECT DISTINCT em.entity_id, e.name, e.entity_type, e.importance
            FROM entity_mentions em
            JOIN entities e ON e.id = em.entity_id
            WHERE em.memory_id IN ({placeholders})""",
        seed_memory_ids
    ).fetchall()

    if not seed_entities:
        return {"entities": [], "edges": [], "expanded_memory_ids": []}

    visited_entities = {r["entity_id"] for r in seed_entities}
    frontier = list(visited_entities)
    all_edges = []
    all_entities = [{
        "id": r["entity_id"], "name": r["name"],
        "entity_type": r["entity_type"], "importance": r["importance"]
    } for r in seed_entities]

    # Step 2: BFS 扩展（最多 max_hops 跳）
    # 截断种子实体到 max_nodes
    if len(all_entities) > max_nodes:
        all_entities = all_entities[:max_nodes]
        visited_entities = {e["id"] for e in all_entities}
        frontier = list(visited_entities)

    for hop in range(max_hops):
        if not frontier or len(visited_entities) >= max_nodes:
            break
        fp = ",".join("?" * len(frontier))
        edges = conn.execute(
            f"""SELECT id, from_entity, to_entity, relation_type, weight, memory_id
                FROM edges
                WHERE from_entity IN ({fp}) OR to_entity IN ({fp})""",
            frontier + frontier
        ).fetchall()

        new_frontier = []
        for edge in edges:
            all_edges.append(dict(edge))
            for eid in [edge["from_entity"], edge["to_entity"]]:
                if eid not in visited_entities and len(visited_entities) < max_nodes:
                    visited_entities.add(eid)
                    new_frontier.append(eid)
                    # 补全实体信息
                    ent = conn.execute(
                        "SELECT id, name, entity_type, importance FROM entities WHERE id=?",
                        (eid,)
                    ).fetchone()
                    if ent:
                        all_entities.append(dict(ent))

        frontier = new_frontier

    # Step 3: 收集扩展发现的额外记忆 ID
    if visited_entities:
        ep = ",".join("?" * len(visited_entities))
        extra_mems = conn.execute(
            f"""SELECT DISTINCT memory_id FROM entity_mentions
                WHERE entity_id IN ({ep})""",
            list(visited_entities)
        ).fetchall()
        expanded_ids = [
            r["memory_id"] for r in extra_mems
            if r["memory_id"] not in seed_memory_ids
        ]
    else:
        expanded_ids = []

    return {
        "entities": all_entities,
        "edges": [dict(e) for e in all_edges],
        "expanded_memory_ids": expanded_ids[:max_nodes],
    }


# ══════════════════════════════════════════════════════════════
# CLI（独立测试用）
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="规则式实体关系提取测试")
    parser.add_argument("--summary", default="alice 决定从1%灰度扩量到5%，bob 负责执行，使用 ai-reco 模型")
    parser.add_argument("--owner", default="alice")
    parser.add_argument("--participants", default="alice,bob")
    parser.add_argument("--tags", default="灰度,ai-reco")
    parser.add_argument("--type", default="conversation", dest="mem_type")
    args = parser.parse_args()

    participants = [p.strip() for p in args.participants.split(",")]
    tags = [t.strip() for t in args.tags.split(",")]

    ents = extract_entities(
        summary=args.summary,
        content={"summary": args.summary},
        tags=tags,
        participants=participants,
        owner=args.owner,
        mem_type=args.mem_type,
    )
    rels = extract_relations(
        entities=ents,
        summary=args.summary,
        content={"summary": args.summary},
        mem_type=args.mem_type,
        owner=args.owner,
        participants=participants,
    )

    print("=== 实体 ===")
    for e in ents:
        print(f"  [{e['entity_type']:8s}] {e['name']}")

    print("\n=== 关系 ===")
    for r in rels:
        print(f"  {r['from_name']} ({r['from_type']}) ──{r['relation_type']}──> "
              f"{r['to_name']} ({r['to_type']})")
