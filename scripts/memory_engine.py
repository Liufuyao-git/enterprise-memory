#!/usr/bin/env python3
"""
memory_engine.py — 企业级长程协作记忆引擎核心
提供：写入、检索、提取、治理四大能力

用法（CLI）：
  python3 memory_engine.py write   --type conversation --content '{"summary":"..."}' --owner alice
  python3 memory_engine.py search  --query "上周灰度决策" --caller alice
  python3 memory_engine.py extract --session-file session.json --participants alice,bob
  python3 memory_engine.py recall  --entity alice --depth 10
  python3 memory_engine.py status
  python3 memory_engine.py consolidate  # 记忆治理（压缩/归档）
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── 路径配置 ─────────────────────────────────────────────────
DEFAULT_DB_PATH = Path.home() / ".openclaw/workspace/memory_engine/memories.db"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

CST = timezone(timedelta(hours=8))


# ══════════════════════════════════════════════════════════════
# 配置加载
# ══════════════════════════════════════════════════════════════

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            import yaml
            with open(CONFIG_PATH) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def get_db_path(config: dict) -> Path:
    raw = config.get("storage", {}).get("db_path", str(DEFAULT_DB_PATH))
    return Path(raw).expanduser()


# ══════════════════════════════════════════════════════════════
# 数据库连接
# ══════════════════════════════════════════════════════════════

def get_conn(config: dict) -> sqlite3.Connection:
    db_path = get_db_path(config)
    if not db_path.exists():
        print(f"[engine] ⚠️  数据库不存在：{db_path}", file=sys.stderr)
        print(f"[engine]    请先运行：python3 scripts/init_db.py", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def is_vector_enabled(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM meta WHERE key='vector_enabled'").fetchone()
    return row and row["value"] == "1"


# ══════════════════════════════════════════════════════════════
# 写入模块
# ══════════════════════════════════════════════════════════════

def content_hash(content: dict) -> str:
    """
    内容哈希，用于去重。
    排除系统自动补充的字段（type/created_at/updated_at/id），
    确保两次传入相同业务内容（无论是否已被 setdefault 补充过元数据）能得到相同哈希。
    """
    exclude_keys = {"created_at", "updated_at", "id", "type"}
    filtered = {k: v for k, v in content.items() if k not in exclude_keys}
    canonical = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def check_duplicate(conn: sqlite3.Connection, hash_val: str) -> bool:
    """检查是否已有相同哈希的记忆"""
    row = conn.execute(
        "SELECT id FROM memories WHERE source_hash=? AND archived=0", (hash_val,)
    ).fetchone()
    return row is not None


def mem_write(
    config: dict,
    mem_type: str,
    content: dict,
    owner: str,
    participants: list = None,
    tags: list = None,
    visibility: str = "team",
    department: str = "",
    summary: str = "",
) -> dict:
    """
    写入一条记忆。
    返回 {"status": "ok"|"duplicate", "id": "..."}
    """
    conn = get_conn(config)
    
    # 自动生成 summary（如果未提供）
    if not summary:
        summary = content.get("summary", "") or content.get("title", "") or str(content)[:200]
    
    participants = participants or []
    tags = tags or []
    now = datetime.now(CST).isoformat()
    mem_id = f"{mem_type[:4]}_{datetime.now(CST).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # 先计算哈希（在补充元数据之前），保证去重基于纯业务内容
    hash_val = content_hash(content)
    if check_duplicate(conn, hash_val):
        conn.close()
        return {"status": "duplicate", "id": None, "message": "相同内容已存在，跳过写入"}
    
    # 补全 content 元数据（哈希计算后才写入这些字段）
    content.setdefault("type", mem_type)
    content.setdefault("created_at", now)
    
    conn.execute(
        """INSERT INTO memories 
           (id, type, content, summary, owner, participants, tags, visibility, 
            department, source_hash, created_at, updated_at, archived)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)""",
        (
            mem_id, mem_type,
            json.dumps(content, ensure_ascii=False),
            summary,
            owner,
            json.dumps(participants, ensure_ascii=False),
            json.dumps(tags, ensure_ascii=False),
            visibility, department, hash_val, now, now,
        )
    )
    
    # 冗余写入专用表
    if mem_type == "event":
        conn.execute(
            "INSERT OR REPLACE INTO events (id, memory_id, title, event_time, participants) VALUES (?,?,?,?,?)",
            (f"evt_{mem_id}", mem_id,
             content.get("title", ""), content.get("time", now),
             json.dumps(participants, ensure_ascii=False))
        )
    elif mem_type == "relationship":
        conn.execute(
            """INSERT OR REPLACE INTO relationships 
               (id, memory_id, person, email, department, role, expertise, last_interaction)
               VALUES (?,?,?,?,?,?,?,?)""",
            (f"rel_{mem_id}", mem_id,
             content.get("person", ""),
             content.get("email", ""),
             content.get("department", department),
             content.get("role", ""),
             json.dumps(content.get("expertise", []), ensure_ascii=False),
             now)
        )
    
    conn.commit()

    # 可选：写入向量库
    if is_vector_enabled(conn):
        _write_vector(conn, mem_id, summary, tags, config)

    # 知识图谱：自动提取实体与关系
    _build_graph(conn, mem_id, mem_type, summary, content, tags, participants, owner, department)

    conn.close()
    return {"status": "ok", "id": mem_id, "message": f"记忆已写入：{mem_id}"}


def _build_graph(conn, memory_id: str, mem_type: str, summary: str,
                 content: dict, tags: list, participants: list,
                 owner: str, department: str):
    """写入记忆后，自动提取实体与关系写入知识图谱表（静默失败，不影响主流程）"""
    try:
        _graph_dir = Path(__file__).parent
        if str(_graph_dir) not in sys.path:
            sys.path.insert(0, str(_graph_dir))
        from graph_extractor import extract_entities, extract_relations, upsert_entities_and_edges

        entities = extract_entities(
            summary=summary, content=content, tags=tags,
            participants=participants, owner=owner,
            mem_type=mem_type, department=department,
        )
        relations = extract_relations(
            entities=entities, summary=summary, content=content,
            mem_type=mem_type, owner=owner, participants=participants,
        )
        upsert_entities_and_edges(conn, memory_id, entities, relations)
    except Exception as e:
        print(f"[engine] ⚠️  图谱构建失败（不影响主流程）：{e}", file=sys.stderr)


def _get_embed_dir(config: dict) -> Path:
    """返回向量库目录（.npy 文件存放位置）"""
    raw = config.get("storage", {}).get(
        "embeddings_path",
        str(Path.home() / ".openclaw/workspace/memory_engine/embeddings")
    )
    return Path(raw).expanduser()


def _get_embedder(config: dict):
    """懒加载全局 Embedder 单例（ONNX，无 torch 依赖）"""
    # embedder.py 与本文件同目录
    embedder_path = Path(__file__).parent
    if str(embedder_path) not in sys.path:
        sys.path.insert(0, str(embedder_path))
    from embedder import get_embedder as _get
    onnx_path = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"
    tok_path  = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2" / "tokenizer.json"
    return _get(onnx_path=onnx_path, tokenizer_path=tok_path)


def _write_vector(conn, mem_id: str, summary: str, tags: list, config: dict):
    """
    写入 numpy 向量库（替换 ChromaDB）。

    存储格式：
      <embeddings_dir>/index.npy       — float32 矩阵 (N, 384)
      <embeddings_dir>/index.ids.json  — mem_id 列表，与矩阵行对应
    """
    try:
        import numpy as np

        embed_dir = _get_embed_dir(config)
        embed_dir.mkdir(parents=True, exist_ok=True)

        mat_path = embed_dir / "index.npy"
        ids_path = embed_dir / "index.ids.json"

        # 加载现有索引
        if mat_path.exists() and ids_path.exists():
            matrix = np.load(str(mat_path))
            with open(ids_path, encoding="utf-8") as f:
                ids = json.load(f)
        else:
            matrix = np.zeros((0, 384), dtype=np.float32)
            ids = []

        # 如果 mem_id 已存在则更新，否则追加
        embedder = _get_embedder(config)
        vec = embedder.encode(summary, normalize=True).reshape(1, -1)  # (1, 384)

        if mem_id in ids:
            idx = ids.index(mem_id)
            matrix[idx] = vec[0]
        else:
            matrix = np.concatenate([matrix, vec], axis=0)
            ids.append(mem_id)

        # 写回磁盘
        np.save(str(mat_path), matrix)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)

    except Exception as e:
        print(f"[engine] ⚠️  向量写入失败（不影响主流程）：{e}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════
# 检索模块
# ══════════════════════════════════════════════════════════════

def mem_search(
    config: dict,
    query: str,
    caller_id: str,
    caller_dept: str = "",
    type_filter: str = None,
    days: int = 90,
    top_k: int = 10,
) -> list:
    """
    双轨检索：FTS5 关键词 + 可选向量语义检索
    自动应用权限过滤。
    """
    conn = get_conn(config)
    
    since = (datetime.now(CST) - timedelta(days=days)).isoformat()
    
    # ── Track A：FTS5 关键词检索 ──────────────────────────────
    type_clause = f"AND m.type = '{type_filter}'" if type_filter else ""
    
    # SQLite FTS5 的 MATCH 查询：
    # 1. `-` 被解析为 NOT 操作符（如 "permission-private" → 含 permission 但不含 private）
    # 2. 某些词（如 private/public）与内置关键字冲突
    # 3. unicode61 分词器对中文按连续字符分词，短语模式要求完整匹配
    # 解决方案：中文查询（含\u4e00-\u9fff）直接用原词；其他用短语模式包裹
    def fts5_escape(q: str) -> str:
        # 如果包含中文字符，直接返回原词（unicode61 会按连续中文分词）
        if any('\u4e00' <= c <= '\u9fff' for c in q):
            return q
        # 英文/数字用短语模式包裹，escape 内部双引号
        q_escaped = q.replace('"', '""')
        return f'"{q_escaped}"'

    safe_query = fts5_escape(query)
    
    fts_rows = conn.execute(
        "SELECT mem_id FROM memories_fts WHERE memories_fts MATCH ?", (safe_query,)
    ).fetchall()

    fts_rows_dict: dict = {}
    if fts_rows:
        hit_ids = [r["mem_id"] for r in fts_rows]
        placeholders = ",".join("?" * len(hit_ids))
        type_sql = f"AND type = '{type_filter}'" if type_filter else ""
        filter_sql = f"""
            SELECT id, type, content, summary, owner, participants, tags, visibility, department, created_at
            FROM memories
            WHERE id IN ({placeholders})
              AND archived = 0
              AND created_at >= ?
              {type_sql}
              AND (
                visibility = 'public'
                OR (visibility = 'team' AND (department = ? OR department = ''))
                OR (visibility = 'private' AND owner = ?)
              )
            LIMIT ?
        """
        # FTS 不限量先拿，权限过滤后用于 Hybrid 排序
        params = hit_ids + [since, caller_dept, caller_id, top_k * 3]
        rows = conn.execute(filter_sql, params).fetchall()
        fts_rows_dict = {r["id"]: dict(r) for r in rows}

    # FTS 无结果且无向量时，快速返回空
    if not fts_rows_dict and not is_vector_enabled(conn):
        conn.close()
        return []

    # ── Track B：向量语义检索 + Hybrid RRF 融合 ────────────────
    if is_vector_enabled(conn):
        vec_results = _vector_search(query, config, top_k * 3)
        # 只保留权限过滤后的向量结果（从 SQLite 补全字段）
        vec_rows_dict: dict = {}
        for vr in vec_results:
            mid = vr["id"]
            if mid not in vec_rows_dict:
                row = conn.execute(
                    """SELECT id, type, content, summary, owner, participants, tags,
                              visibility, department, created_at
                       FROM memories
                       WHERE id=? AND archived=0 AND created_at>=?
                         AND (
                           visibility='public'
                           OR (visibility='team' AND (department=? OR department=''))
                           OR (visibility='private' AND owner=?)
                         )""",
                    (mid, since, caller_dept, caller_id)
                ).fetchone()
                if row:
                    vec_rows_dict[mid] = dict(row)

        # RRF 融合（k=60）
        all_ids = list({**fts_rows_dict, **vec_rows_dict}.keys())
        fts_rank = {mid: rank for rank, mid in enumerate(fts_rows_dict.keys(), 1)}
        vec_rank = {mid: rank for rank, mid in enumerate(vec_rows_dict.keys(), 1)}
        rrf_k = 60
        rrf_scores = {
            mid: (1.0 / (rrf_k + fts_rank.get(mid, len(all_ids) + rrf_k)) +
                  1.0 / (rrf_k + vec_rank.get(mid, len(all_ids) + rrf_k)))
            for mid in all_ids
        }
        ranked_ids = sorted(all_ids, key=lambda x: -rrf_scores[x])[:top_k]
        # 补全所有行
        all_rows = {**fts_rows_dict, **vec_rows_dict}
        final_rows = [all_rows[mid] for mid in ranked_ids if mid in all_rows]
    else:
        # 纯 FTS 模式
        final_rows = list(fts_rows_dict.values())[:top_k]

    conn.close()

    # 格式化输出
    return [_format_memory(r) for r in final_rows]


def _vector_search(query: str, config: dict, top_k: int) -> list:
    """
    numpy 向量检索（替换 ChromaDB）。
    返回 [{"id": mem_id, "score": cosine_sim}, ...]，按相似度降序。
    """
    try:
        import numpy as np

        embed_dir = _get_embed_dir(config)
        mat_path = embed_dir / "index.npy"
        ids_path = embed_dir / "index.ids.json"

        if not mat_path.exists() or not ids_path.exists():
            return []

        matrix = np.load(str(mat_path))  # (N, 384)
        with open(ids_path, encoding="utf-8") as f:
            ids = json.load(f)

        if len(ids) == 0:
            return []

        embedder = _get_embedder(config)
        q_vec = embedder.encode(query, normalize=True)  # (384,)
        scores = matrix @ q_vec                          # (N,)

        top_n = min(top_k, len(ids))
        top_indices = scores.argsort()[::-1][:top_n]
        return [{"id": ids[i], "score": float(scores[i])} for i in top_indices]

    except Exception as e:
        print(f"[engine] ⚠️  向量检索失败，降级为纯 FTS5：{e}", file=sys.stderr)
        return []


def mem_recall(
    config: dict,
    entity: str,
    depth: int = 10,
    caller_id: str = "",
    caller_dept: str = "",
) -> dict:
    """查询某人/某 tag 的完整记忆链"""
    conn = get_conn(config)
    
    rows = conn.execute(
        """SELECT id, type, summary, owner, participants, tags, created_at, visibility, department
           FROM memories
           WHERE (
               owner = ?
               OR participants LIKE ?
               OR tags LIKE ?
           )
           AND archived = 0
           AND (
               visibility = 'public'
               OR (visibility = 'team' AND (department = ? OR department = ''))
               OR (visibility = 'private' AND owner = ?)
           )
           ORDER BY created_at DESC
           LIMIT ?""",
        (entity, f'%{entity}%', f'%{entity}%', caller_dept, caller_id, depth)
    ).fetchall()
    
    conn.close()
    
    memories = [_format_memory(dict(r)) for r in rows]
    return {
        "entity": entity,
        "total": len(memories),
        "memories": memories,
    }


def mem_graph_search(
    config: dict,
    query: str,
    caller_id: str,
    caller_dept: str = "",
    days: int = 90,
    top_k: int = 10,
    max_hops: int = 1,
) -> dict:
    """
    图增强检索：先 Hybrid 检索，再沿知识图谱扩展一跳，补充关联记忆。

    步骤：
      1. Hybrid 检索得到 seed 记忆（top_k 条）
      2. 从 seed 出发，BFS 扩展 max_hops 跳，找到关联实体
      3. 通过关联实体找到额外记忆
      4. 合并 seed + 扩展记忆，按来源标注返回

    返回：
    {
      "direct": [...],     # Hybrid 直接命中的记忆
      "expanded": [...],   # 图扩展发现的关联记忆
      "entities": [...],   # 途经的实体节点
      "edges": [...],      # 途经的关系边
      "total": N,
    }
    """
    # Step 1: Hybrid 检索
    direct = mem_search(
        config=config, query=query, caller_id=caller_id,
        caller_dept=caller_dept, days=days, top_k=top_k,
    )
    seed_ids = [m["id"] for m in direct]

    if not seed_ids:
        return {"direct": [], "expanded": [], "entities": [], "edges": [], "total": 0}

    conn = get_conn(config)
    since = (datetime.now(CST) - timedelta(days=days)).isoformat()

    try:
        from graph_extractor import graph_bfs
    except ImportError:
        conn.close()
        return {"direct": direct, "expanded": [], "entities": [], "edges": [], "total": len(direct)}

    # Step 2: BFS 扩展
    graph_result = graph_bfs(conn, seed_ids, max_hops=max_hops, max_nodes=50)
    expanded_ids = graph_result["expanded_memory_ids"]

    # Step 3: 取扩展记忆，做权限过滤
    expanded = []
    if expanded_ids:
        ph = ",".join("?" * len(expanded_ids))
        rows = conn.execute(
            f"""SELECT id, type, summary, owner, participants, tags,
                       visibility, department, created_at
                FROM memories
                WHERE id IN ({ph})
                  AND archived = 0
                  AND created_at >= ?
                  AND (
                    visibility = 'public'
                    OR (visibility = 'team' AND (department = ? OR department = ''))
                    OR (visibility = 'private' AND owner = ?)
                  )
                LIMIT ?""",
            expanded_ids + [since, caller_dept, caller_id, top_k]
        ).fetchall()
        expanded = [_format_memory(dict(r)) for r in rows
                    if r["id"] not in {m["id"] for m in direct}]

    conn.close()

    return {
        "direct": direct,
        "expanded": expanded,
        "entities": graph_result["entities"],
        "edges": graph_result["edges"],
        "total": len(direct) + len(expanded),
    }


def mem_graph_recall(
    config: dict,
    entity_name: str,
    entity_type: str = None,
    caller_id: str = "",
    caller_dept: str = "",
    depth: int = 2,
    days: int = 365,
) -> dict:
    """
    实体邻居查询：以实体为中心，返回所有关联实体、关系边和相关记忆。

    用途：
      - "alice 参与了哪些项目？"
      - "ai-reco 项目涉及哪些人和决策？"
      - "灰度发布和哪些实体相关？"
    """
    conn = get_conn(config)
    since = (datetime.now(CST) - timedelta(days=days)).isoformat()

    # 1. 查找实体
    if entity_type:
        row = conn.execute(
            "SELECT id, name, entity_type, importance FROM entities WHERE name=? AND entity_type=?",
            (entity_name, entity_type)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, name, entity_type, importance FROM entities WHERE name=?",
            (entity_name,)
        ).fetchone()

    if not row:
        # 尝试模糊匹配
        row = conn.execute(
            "SELECT id, name, entity_type, importance FROM entities WHERE name LIKE ?",
            (f"%{entity_name}%",)
        ).fetchone()

    if not row:
        conn.close()
        return {
            "entity": entity_name,
            "found": False,
            "neighbors": [],
            "edges": [],
            "memories": [],
        }

    entity_id = row["id"]

    # 直接查该实体的邻居边
    edges_out = conn.execute(
        """SELECT e.id, e.from_entity, e.to_entity, e.relation_type, e.weight,
                  ef.name as from_name, ef.entity_type as from_type,
                  et.name as to_name, et.entity_type as to_type
           FROM edges e
           JOIN entities ef ON ef.id = e.from_entity
           JOIN entities et ON et.id = e.to_entity
           WHERE e.from_entity = ? OR e.to_entity = ?
           ORDER BY e.weight DESC
           LIMIT 50""",
        (entity_id, entity_id)
    ).fetchall()

    # 3. 邻居实体
    neighbor_ids = set()
    edges_info = []
    for edge in edges_out:
        edges_info.append(dict(edge))
        neighbor_ids.add(edge["from_entity"])
        neighbor_ids.add(edge["to_entity"])
    neighbor_ids.discard(entity_id)

    neighbors = []
    for nid in neighbor_ids:
        n = conn.execute(
            "SELECT id, name, entity_type, importance FROM entities WHERE id=?", (nid,)
        ).fetchone()
        if n:
            neighbors.append(dict(n))

    # 4. 该实体相关记忆（权限过滤）
    mem_rows = conn.execute(
        """SELECT m.id, m.type, m.summary, m.owner, m.participants,
                  m.tags, m.visibility, m.department, m.created_at
           FROM entity_mentions em
           JOIN memories m ON m.id = em.memory_id
           WHERE em.entity_id = ?
             AND m.archived = 0
             AND m.created_at >= ?
             AND (
               m.visibility = 'public'
               OR (m.visibility = 'team' AND (m.department = ? OR m.department = ''))
               OR (m.visibility = 'private' AND m.owner = ?)
             )
           ORDER BY m.created_at DESC
           LIMIT 20""",
        (entity_id, since, caller_dept, caller_id)
    ).fetchall()

    memories = [_format_memory(dict(r)) for r in mem_rows]

    conn.close()

    return {
        "entity": {"id": entity_id, "name": row["name"],
                   "entity_type": row["entity_type"],
                   "importance": row["importance"]},
        "found": True,
        "neighbors": sorted(neighbors, key=lambda x: -x["importance"]),
        "edges": edges_info,
        "memories": memories,
    }


def _format_memory(row: dict) -> dict:
    """格式化记忆记录，解析 JSON 字段"""
    result = dict(row)
    for field in ["participants", "tags"]:
        if isinstance(result.get(field), str):
            try:
                result[field] = json.loads(result[field])
            except Exception:
                pass
    if isinstance(result.get("content"), str):
        try:
            result["content"] = json.loads(result["content"])
        except Exception:
            pass
    return result


# ══════════════════════════════════════════════════════════════
# 自动提取模块
# ══════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """你是企业记忆提取器。从以下对话中提取结构化记忆。

参与者：{participants}
对话内容：
{conversation_text}

请严格按如下 JSON 格式输出，不要添加任何其他文字：
{{
  "should_store": true/false,
  "reason": "是否值得存储的简短原因（中文）",
  "records": [
    {{
      "type": "conversation",
      "summary": "一句话摘要（中文，50字以内）",
      "decisions": ["决策1", "决策2"],
      "action_items": [{{"owner": "人名", "task": "任务", "due": "YYYY-MM-DD或空"}}],
      "tags": ["标签1", "标签2"],
      "visibility": "public/team/private"
    }}
  ]
}}

判断是否值得存储的标准：
1. 包含明确决策（"确认"、"决定"、"方案是"、"约定"）
2. 包含 Action Item（有 owner 的任务）
3. 多人参与的跨部门讨论
4. 重要里程碑或风险点

如果只是闲聊或没有实质内容，should_store=false，records=[]"""


def mem_extract(
    config: dict,
    conversation_text: str,
    participants: list,
    owner: str = "",
    department: str = "",
    call_llm_fn=None,
) -> dict:
    """
    从对话文本自动提取并写入记忆。
    call_llm_fn: 可注入的 LLM 调用函数，签名 fn(prompt: str) -> str
                 为 None 时，输出提示词供外部调用
    """
    # 至少 2 人才触发自动提取
    min_participants = config.get("extraction", {}).get("min_participants", 2)
    if len(participants) < min_participants:
        return {
            "status": "skipped",
            "reason": f"参与人数 {len(participants)} < 触发阈值 {min_participants}"
        }
    
    prompt = EXTRACT_PROMPT.format(
        participants=", ".join(participants),
        conversation_text=conversation_text[:4000],  # 限制长度
    )
    
    if call_llm_fn is None:
        # 无 LLM 时，返回提示词供外部使用
        return {
            "status": "pending_llm",
            "prompt": prompt,
            "message": "请将 prompt 传给 LLM，将输出 JSON 传回 mem_extract_commit()"
        }
    
    # 调用 LLM
    try:
        raw_output = call_llm_fn(prompt)
        return mem_extract_commit(config, raw_output, participants, owner, department)
    except Exception as e:
        return {"status": "error", "message": str(e)}


def mem_extract_commit(
    config: dict,
    llm_output: str,
    participants: list,
    owner: str = "",
    department: str = "",
) -> dict:
    """将 LLM 输出的 JSON 提交写入记忆库"""
    # 提取 JSON（容错：可能有多余的 markdown 代码块）
    text = llm_output.strip()
    if "```" in text:
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1).strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"LLM 输出解析失败：{e}", "raw": llm_output}

    # 防御非 dict 类型（如 [] 或 null）
    if not isinstance(data, dict):
        return {"status": "error", "message": f"LLM 输出应为 JSON 对象，实际为 {type(data).__name__}",
                "raw": llm_output}

    if not data.get("should_store", False):
        return {"status": "skipped", "reason": data.get("reason", "LLM 判断不值得存储")}
    
    written = []
    for record in (data.get("records") or []):
        result = mem_write(
            config=config,
            mem_type=record.get("type", "conversation"),
            content=record,
            owner=owner or (participants[0] if participants else ""),
            participants=participants,
            tags=record.get("tags", []),
            visibility=record.get("visibility", "team"),
            department=department,
            summary=record.get("summary", ""),
        )
        written.append(result)
    
    return {
        "status": "ok",
        "extracted": len(written),
        "records": written,
    }


# ══════════════════════════════════════════════════════════════
# 记忆治理模块
# ══════════════════════════════════════════════════════════════

def mem_status(config: dict) -> dict:
    """查询记忆库统计信息"""
    conn = get_conn(config)
    
    total = conn.execute("SELECT COUNT(*) as n FROM memories WHERE archived=0").fetchone()["n"]
    by_type = conn.execute(
        "SELECT type, COUNT(*) as n FROM memories WHERE archived=0 GROUP BY type"
    ).fetchall()
    oldest = conn.execute(
        "SELECT MIN(created_at) as t FROM memories WHERE archived=0"
    ).fetchone()["t"]
    newest = conn.execute(
        "SELECT MAX(created_at) as t FROM memories WHERE archived=0"
    ).fetchone()["t"]
    archived_count = conn.execute("SELECT COUNT(*) as n FROM memories WHERE archived=1").fetchone()["n"]
    vector_enabled = is_vector_enabled(conn)
    
    conn.close()
    
    return {
        "total_active": total,
        "archived": archived_count,
        "by_type": {row["type"]: row["n"] for row in by_type},
        "oldest_record": oldest,
        "newest_record": newest,
        "vector_enabled": vector_enabled,
        "db_path": str(get_db_path(config)),
    }


def mem_consolidate(config: dict, dry_run: bool = False) -> dict:
    """
    记忆治理：归档超期记忆，压缩活跃层
    
    分层策略：
      新鲜层（0-7天）  → 不动
      活跃层（7-30天） → 标记为待压缩（后续版本实现 LLM 合并）
      沉淀层（30-90天）→ 不动（已在活跃层压缩过）
      归档层（90天+）  → archived=1
    """
    retention = config.get("retention", {})
    archive_days = retention.get("archive_days", 90)
    
    archive_before = (datetime.now(CST) - timedelta(days=archive_days)).isoformat()
    
    conn = get_conn(config)
    
    # 查询待归档数量
    to_archive = conn.execute(
        "SELECT COUNT(*) as n FROM memories WHERE archived=0 AND created_at < ?",
        (archive_before,)
    ).fetchone()["n"]
    
    result = {
        "archive_before": archive_before,
        "to_archive": to_archive,
        "archived": 0,
        "dry_run": dry_run,
    }
    
    if not dry_run and to_archive > 0:
        conn.execute(
            "UPDATE memories SET archived=1, updated_at=? WHERE archived=0 AND created_at < ?",
            (datetime.now(CST).isoformat(), archive_before)
        )
        conn.commit()
        result["archived"] = to_archive
        print(f"[engine] 已归档 {to_archive} 条超过 {archive_days} 天的记忆")
    
    conn.close()
    return result


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="企业记忆引擎 CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # write
    p_write = sub.add_parser("write", help="写入记忆")
    p_write.add_argument("--type", required=True, choices=["conversation", "document", "relationship", "event"])
    p_write.add_argument("--content", required=True, help="JSON 字符串或文件路径")
    p_write.add_argument("--owner", required=True)
    p_write.add_argument("--participants", default="", help="逗号分隔")
    p_write.add_argument("--tags", default="", help="逗号分隔")
    p_write.add_argument("--visibility", default="team", choices=["public", "team", "private"])
    p_write.add_argument("--department", default="")
    p_write.add_argument("--summary", default="")
    
    # search
    p_search = sub.add_parser("search", help="检索记忆")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--caller", required=True)
    p_search.add_argument("--dept", default="")
    p_search.add_argument("--type", default=None, dest="type_filter")
    p_search.add_argument("--days", type=int, default=90)
    p_search.add_argument("--top-k", type=int, default=10)
    
    # recall
    p_recall = sub.add_parser("recall", help="查询某人/某 tag 的记忆链")
    p_recall.add_argument("--entity", required=True)
    p_recall.add_argument("--depth", type=int, default=10)
    p_recall.add_argument("--caller", default="")
    p_recall.add_argument("--dept", default="")
    
    # extract
    p_extract = sub.add_parser("extract", help="从对话文本自动提取记忆（输出提示词）")
    p_extract.add_argument("--text", help="对话文本（直接输入）")
    p_extract.add_argument("--file", help="对话文本文件路径")
    p_extract.add_argument("--participants", required=True, help="逗号分隔")
    p_extract.add_argument("--owner", default="")
    p_extract.add_argument("--dept", default="")
    
    # commit（配合 extract 使用：将 LLM 输出提交写入）
    p_commit = sub.add_parser("commit", help="提交 LLM 提取结果")
    p_commit.add_argument("--llm-output", help="LLM 输出 JSON 字符串")
    p_commit.add_argument("--llm-file", help="LLM 输出文件路径")
    p_commit.add_argument("--participants", required=True)
    p_commit.add_argument("--owner", default="")
    p_commit.add_argument("--dept", default="")
    
    # graph-search
    p_gsearch = sub.add_parser("graph-search", help="图增强检索（Hybrid + 图扩展）")
    p_gsearch.add_argument("--query", required=True)
    p_gsearch.add_argument("--caller", required=True)
    p_gsearch.add_argument("--dept", default="")
    p_gsearch.add_argument("--days", type=int, default=90)
    p_gsearch.add_argument("--top-k", type=int, default=10)
    p_gsearch.add_argument("--hops", type=int, default=1)

    # graph-recall
    p_grecall = sub.add_parser("graph-recall", help="实体邻居查询")
    p_grecall.add_argument("--entity", required=True)
    p_grecall.add_argument("--entity-type", default=None)
    p_grecall.add_argument("--caller", default="")
    p_grecall.add_argument("--dept", default="")
    p_grecall.add_argument("--depth", type=int, default=2)
    p_grecall.add_argument("--days", type=int, default=365)

    # status
    sub.add_parser("status", help="查看记忆库统计")
    
    # consolidate
    p_cons = sub.add_parser("consolidate", help="记忆治理（归档）")
    p_cons.add_argument("--dry-run", action="store_true")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.cmd == "write":
        # 解析 content
        content_str = args.content
        if content_str.startswith("@"):
            content_str = Path(content_str[1:]).read_text()
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {"summary": content_str}
        
        result = mem_write(
            config=config,
            mem_type=args.type,
            content=content,
            owner=args.owner,
            participants=[p.strip() for p in args.participants.split(",") if p.strip()],
            tags=[t.strip() for t in args.tags.split(",") if t.strip()],
            visibility=args.visibility,
            department=args.department,
            summary=args.summary,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.cmd == "search":
        results = mem_search(
            config=config,
            query=args.query,
            caller_id=args.caller,
            caller_dept=args.dept,
            type_filter=args.type_filter,
            days=args.days,
            top_k=args.top_k,
        )
        print(json.dumps({"total": len(results), "results": results}, ensure_ascii=False, indent=2))
    
    elif args.cmd == "recall":
        result = mem_recall(
            config=config,
            entity=args.entity,
            depth=args.depth,
            caller_id=args.caller,
            caller_dept=args.dept,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.cmd == "extract":
        text = args.text or ""
        if args.file:
            text = Path(args.file).read_text(encoding="utf-8")
        participants = [p.strip() for p in args.participants.split(",") if p.strip()]
        result = mem_extract(
            config=config,
            conversation_text=text,
            participants=participants,
            owner=args.owner,
            department=args.dept,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.cmd == "commit":
        llm_output = args.llm_output or ""
        if args.llm_file:
            llm_output = Path(args.llm_file).read_text(encoding="utf-8")
        participants = [p.strip() for p in args.participants.split(",") if p.strip()]
        result = mem_extract_commit(
            config=config,
            llm_output=llm_output,
            participants=participants,
            owner=args.owner,
            department=args.dept,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.cmd == "graph-search":
        result = mem_graph_search(
            config=config, query=args.query,
            caller_id=args.caller, caller_dept=args.dept,
            days=args.days, top_k=args.top_k, max_hops=args.hops,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.cmd == "graph-recall":
        result = mem_graph_recall(
            config=config, entity_name=args.entity,
            entity_type=args.entity_type,
            caller_id=args.caller, caller_dept=args.dept,
            depth=args.depth, days=args.days,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.cmd == "status":
        result = mem_status(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.cmd == "consolidate":
        result = mem_consolidate(config, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
