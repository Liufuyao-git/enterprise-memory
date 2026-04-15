---
name: enterprise-memory
version: 1.0.0
updated_at: "2026-04-14"
description: >
  企业级长程协作记忆引擎。为 OpenClaw 智能体提供跨部门、跨周级的记忆管理能力。
  支持四类记忆（对话、文档、人际、事件），自动提取决策与待办，语义/关键词双轨检索，
  多用户权限隔离。零依赖安装（SQLite），可选向量库扩展。
---

# enterprise-memory — 企业级长程协作记忆引擎

## 功能概述

解决企业跨部门协作中 AI "失忆" 问题：
- **会话断层** → 跨会话持久化记忆
- **跨人断层** → 多用户共享记忆库 + 权限隔离
- **时间衰减** → 周～月级记忆治理（归档/压缩）

**四类记忆**：
| 类型 | 场景 | 自动触发条件 |
|------|------|-------------|
| conversation | 会议、讨论、决策 | 多人会话结束 |
| document | 文档、报告、资源 | 显式记录 |
| relationship | 人员、部门、职责 | 首次协作 |
| event | 里程碑、评审、发布 | 显式记录 |

## 安装

### 零依赖模式（推荐）

```bash
# 1. 复制 Skill 到 workspace
cp -r enterprise-memory ~/.openclaw/workspace/skills/

# 2. 初始化数据库
python3 ~/.openclaw/workspace/skills/enterprise-memory/scripts/init_db.py

# 3. 验证
python3 ~/.openclaw/workspace/skills/enterprise-memory/scripts/memory_engine.py status
```

### 向量检索扩展（可选）

```bash
pip install onnxruntime numpy
python3 scripts/init_db.py --with-vectors
```

## 触发词

**写入类**：
- "记录刚才的决策"
- "把这次讨论存到记忆库"
- "记住 xxx 负责 xxx"

**检索类**：
- "找一下上周和产品侧的决策"
- "xxx 上次说的方案是什么"
- "3 月份关于灰度比例的约定"

**人际类**：
- "xxx 是负责什么的"
- "和推荐组的协作记录"

**状态类**：
- "记忆库里有多少条记录"
- "最近一个月存了什么"

## 配置

```yaml
# ~/.openclaw/workspace/skills/enterprise-memory/config.yaml
storage:
  backend: sqlite
  db_path: ~/.openclaw/workspace/memory_engine/memories.db
  embeddings_path: ~/.openclaw/workspace/memory_engine/embeddings/

retention:
  fresh_days: 7
  active_days: 30
  archive_days: 90

extraction:
  auto_extract: true
  min_participants: 2

multi_user:
  enabled: true
  default_visibility: team

search:
  hybrid_alpha: 0.6
  top_k: 10
```

## 权限模型

| visibility | 可见范围 |
|------------|---------|
| public | 所有人 |
| team | 同 department |
| private | 仅 owner |

## 架构白皮书

详见 `docs/whitepaper.md`

## 评测报告

详见 `docs/evaluation.md`

## 文件目录

```
enterprise-memory/
├── SKILL.md                    # 本文件
├── config.yaml                 # 配置文件
├── docs/
│   ├── whitepaper.md          # 架构白皮书
│   └── evaluation.md          # 评测报告
├── scripts/
│   ├── init_db.py             # 数据库初始化
│   └── memory_engine.py       # 核心引擎（写入/检索/提取/治理）
└── tests/
    ├── run_tests.py           # 评测用例
    └── test_report.json       # 评测结果
```

## 与 context-slim 的关系

| | context-slim | enterprise-memory |
|--|--------------|-------------------|
| 目标 | 精简上下文 token | 增强长程记忆 |
| 方向 | 删 | 存 |
| 用户 | 单人 | 多人共享 |
| 触发 | 手动 | 自动 + 手动 |

**推荐组合**：
- `enterprise-memory` 存入重要记忆
- `context-slim` 定期清理 MEMORY.md
- `memoryres/` 作为冷存储

## 注意事项

1. **首次使用需初始化**：必须先运行 `init_db.py`
2. **自动提取依赖 LLM**：跨部门会话（≥2 人）结束后自动触发，需 LLM 支持
3. **FTS5 查询转义**：内部已处理 FTS5 特殊字符（如 `-`），无需用户干预
4. **凭据保护**：`private` 记忆不会被其他用户检索到

## 作者

- **项目**：企业级长程协作记忆引擎
- **作者**：liufuyao
- **日期**：2026-04-14
