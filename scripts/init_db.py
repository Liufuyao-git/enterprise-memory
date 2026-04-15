#!/usr/bin/env python3
"""
init_db.py — 企业记忆引擎数据库初始化
用法：
  python3 init_db.py                  # SQLite 模式（零依赖）
  python3 init_db.py --with-vectors   # 同时启用 ChromaDB 向量库
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB_DIR = Path.home() / ".openclaw" / "workspace" / "memory_engine"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "memories.db"
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def get_db_path(config: dict) -> Path:
    raw = config.get("storage", {}).get("db_path", str(DEFAULT_DB_PATH))
    return Path(raw).expanduser()


def init_sqlite(db_path: Path):
    """创建 SQLite 数据库 + FTS5 虚拟表"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ── 主表 ──────────────────────────────────────────────────
    c.executescript("""
    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        type        TEXT NOT NULL,           -- conversation/document/relationship/event
        content     TEXT NOT NULL,           -- JSON blob（完整结构化记忆）
        summary     TEXT NOT NULL DEFAULT '',-- 用于 FTS 的纯文本摘要
        owner       TEXT NOT NULL DEFAULT '',-- 记录创建人
        participants TEXT NOT NULL DEFAULT '[]', -- JSON 数组
        tags        TEXT NOT NULL DEFAULT '[]',  -- JSON 数组
        visibility  TEXT NOT NULL DEFAULT 'team',-- public/team/private
        department  TEXT NOT NULL DEFAULT '',
        source_hash TEXT NOT NULL DEFAULT '',-- SHA256，用于去重
        created_at  TEXT NOT NULL,           -- ISO8601
        updated_at  TEXT NOT NULL,
        archived    INTEGER NOT NULL DEFAULT 0
    );

    -- 事件时间线表（从 event 类记忆冗余存储，便于按时间范围查询）
    CREATE TABLE IF NOT EXISTS events (
        id          TEXT PRIMARY KEY,
        memory_id   TEXT NOT NULL REFERENCES memories(id),
        title       TEXT NOT NULL,
        event_time  TEXT NOT NULL,           -- ISO8601
        participants TEXT NOT NULL DEFAULT '[]'
    );

    -- 人际关系表（从 relationship 类记忆冗余存储）
    CREATE TABLE IF NOT EXISTS relationships (
        id          TEXT PRIMARY KEY,
        memory_id   TEXT NOT NULL REFERENCES memories(id),
        person      TEXT NOT NULL,
        email       TEXT NOT NULL DEFAULT '',
        department  TEXT NOT NULL DEFAULT '',
        role        TEXT NOT NULL DEFAULT '',
        expertise   TEXT NOT NULL DEFAULT '[]',
        last_interaction TEXT NOT NULL DEFAULT ''
    );

    -- FTS5 全文检索虚拟表（独立表，通过 mem_id 与主表关联，避免 content= 模式的列名歧义）
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        mem_id UNINDEXED,
        summary,
        tags,
        tokenize='unicode61 remove_diacritics 1'
    );

    -- 写入触发器：同步更新 FTS（tags JSON 数组转为空格分隔字符串便于检索）
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(mem_id, summary, tags)
        VALUES (new.id, new.summary, replace(replace(replace(new.tags, '[', ''), ']', ''), '"', ' '));
    END;

    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
        DELETE FROM memories_fts WHERE mem_id = old.id;
    END;

    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
        DELETE FROM memories_fts WHERE mem_id = old.id;
        INSERT INTO memories_fts(mem_id, summary, tags)
        VALUES (new.id, new.summary, replace(replace(replace(new.tags, '[', ''), ']', ''), '"', ' '));
    END;

    -- 元数据表
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    -- ══════════════════════════════════════════════
    -- 知识图谱扩展（v1.1）
    -- ══════════════════════════════════════════════

    -- 实体节点表
    -- entity_type: person / project / decision / concept / org
    CREATE TABLE IF NOT EXISTS entities (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        aliases      TEXT NOT NULL DEFAULT '[]',
        entity_type  TEXT NOT NULL DEFAULT 'concept',
        department   TEXT NOT NULL DEFAULT '',
        description  TEXT NOT NULL DEFAULT '',
        importance   REAL NOT NULL DEFAULT 1.0,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_type
        ON entities(name, entity_type);

    -- 关系边表（有向图）
    -- relation_type: 执行/负责/参与/决定/属于/上线/依赖/提出/关联
    CREATE TABLE IF NOT EXISTS edges (
        id            TEXT PRIMARY KEY,
        from_entity   TEXT NOT NULL REFERENCES entities(id),
        to_entity     TEXT NOT NULL REFERENCES entities(id),
        relation_type TEXT NOT NULL,
        memory_id     TEXT NOT NULL REFERENCES memories(id),
        weight        REAL NOT NULL DEFAULT 1.0,
        created_at    TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_entity);
    CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_entity);
    CREATE INDEX IF NOT EXISTS idx_edges_mem  ON edges(memory_id);

    -- 记忆-实体关联表（一条记忆可涉及多个实体）
    CREATE TABLE IF NOT EXISTS entity_mentions (
        id         TEXT PRIMARY KEY,
        memory_id  TEXT NOT NULL REFERENCES memories(id),
        entity_id  TEXT NOT NULL REFERENCES entities(id),
        role       TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_mentions_mem ON entity_mentions(memory_id);
    CREATE INDEX IF NOT EXISTS idx_mentions_ent ON entity_mentions(entity_id);
    """)

    # 写入初始元数据
    c.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', '1.1')")
    c.execute("INSERT OR REPLACE INTO meta VALUES ('created_at', datetime('now'))")
    c.execute("INSERT OR REPLACE INTO meta VALUES ('vector_enabled', '0')")

    conn.commit()
    conn.close()
    print(f"[init] SQLite 数据库初始化完成：{db_path}")


def init_vectors(config: dict):
    """
    初始化 numpy 向量库目录（替换 ChromaDB，零外部依赖）。
    只需确认 ONNX 模型文件存在，并创建 embeddings 目录。
    """
    import numpy as np

    raw_path = config.get("storage", {}).get(
        "embeddings_path",
        str(Path.home() / ".openclaw/workspace/memory_engine/embeddings")
    )
    embed_path = Path(raw_path).expanduser()
    embed_path.mkdir(parents=True, exist_ok=True)

    # 验证 ONNX 模型文件
    onnx_path = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"
    tok_path  = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2" / "tokenizer.json"
    if not onnx_path.exists():
        print(f"[init] ⚠️  ONNX 模型文件不存在：{onnx_path}", file=sys.stderr)
        print(f"[init]    请先将模型文件放入 models/all-MiniLM-L6-v2/onnx/model.onnx", file=sys.stderr)
        return False
    if not tok_path.exists():
        print(f"[init] ⚠️  tokenizer.json 不存在：{tok_path}", file=sys.stderr)
        return False

    # 创建空索引文件（0条记录）
    mat_path = embed_path / "index.npy"
    ids_path = embed_path / "index.ids.json"
    if not mat_path.exists():
        np.save(str(mat_path), np.zeros((0, 384), dtype=np.float32))
        print(f"[init] 创建空向量矩阵：{mat_path}")
    if not ids_path.exists():
        with open(ids_path, "w") as f:
            json.dump([], f)
        print(f"[init] 创建空 ID 列表：{ids_path}")

    print(f"[init] numpy 向量库初始化完成：{embed_path}")
    print(f"[init] ONNX 模型：{onnx_path}")

    # 更新 SQLite 元数据
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('vector_enabled', '1')")
    conn.execute(f"INSERT OR REPLACE INTO meta VALUES ('embeddings_path', '{embed_path}')")
    conn.commit()
    conn.close()
    return True


def load_config() -> dict:
    """加载 config.yaml，不存在则返回默认值"""
    if DEFAULT_CONFIG_PATH.exists():
        try:
            import yaml
            with open(DEFAULT_CONFIG_PATH) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
        except Exception as e:
            print(f"[init] 警告：读取 config.yaml 失败（{e}），使用默认配置", file=sys.stderr)
    return {}


def main():
    parser = argparse.ArgumentParser(description="企业记忆引擎 — 数据库初始化")
    parser.add_argument("--with-vectors", action="store_true", help="同时初始化 ChromaDB 向量库")
    parser.add_argument("--db-path", type=str, default=None, help="指定 SQLite 数据库路径")
    args = parser.parse_args()

    config = load_config()
    db_path = Path(args.db_path).expanduser() if args.db_path else get_db_path(config)

    print(f"[init] 企业记忆引擎初始化")
    print(f"[init] 数据库路径：{db_path}")

    init_sqlite(db_path)

    if args.with_vectors:
        success = init_vectors(config)
        if success:
            print("[init] ✅ 向量检索已启用（ChromaDB）")
        else:
            print("[init] ⚠️  向量检索未启用，将使用纯 SQLite FTS5 关键词检索")
    else:
        print("[init] ℹ️  向量检索未启用（使用 --with-vectors 开启）")

    print("[init] ✅ 初始化完成")


if __name__ == "__main__":
    main()
