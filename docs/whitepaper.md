# 企业级长程协作记忆引擎 — 架构白皮书

**版本**：v1.1  
**日期**：2026-04-15  
**作者**：旺财（基于 OpenClaw 平台）

---

## 一、问题定义

### 1.1 核心痛点

企业跨部门协作场景中，AI 智能体面临三类系统性"失忆"问题：

| 失忆类型 | 表现 | 影响 |
|---------|------|------|
| **会话断层** | 跨会话无法记住上次决策 | 重复沟通、反复确认 |
| **跨人断层** | 不同部门的 AI 实例信息孤岛 | 协作摩擦、信息不一致 |
| **时间衰减** | 数周前的决策、约定无法检索 | 遗忘性错误、追责困难 |

### 1.2 现有方案局限

| 方案 | 局限 |
|------|------|
| MEMORY.md 文件 | 单人、单机、无检索、无结构 |
| context-slim | 瘦身工具，不解决跨人/跨时问题 |
| MemGPT / A-MEM | 个人助手场景，无企业多用户隔离 |
| RAG 知识库 | 静态文档，无主动写入、无人际图谱 |

### 1.3 目标定义

构建一个**独立的记忆中间层**，具备：
- 四类记忆（对话、文档、人际、事件）统一管理
- 周～月级长程持久化
- 多用户共享 + 权限隔离
- 自动触发写入（跨部门协作场景识别）
- 语义检索（自然语言查询历史决策）
- 以 OpenClaw Skill 形式交付，零外部依赖即可运行

---

## 二、记忆类型定义

企业办公场景中，"记忆"分为四个正交维度：

### 2.1 对话记忆（Conversational Memory）

**定义**：多轮会话中产生的决策、共识、待办、疑问。

**结构**：
```json
{
  "type": "conversation",
  "id": "conv_20260414_001",
  "participants": ["user_a", "user_b"],
  "timestamp": "2026-04-14T19:30:00+08:00",
  "summary": "确认 AI 推荐决策方案进入灰度阶段",
  "decisions": ["灰度比例 1% 起，观察 48h 后扩量"],
  "action_items": [{"owner": "user_a", "task": "部署 20260414 版本", "due": "2026-04-15"}],
  "tags": ["ai-reco", "灰度", "推荐决策"],
  "source_session": "session:abc123"
}
```

**自动触发条件**：
- 检测到决策性语言（"确认"、"决定"、"方案是"、"我们约定"）
- 检测到 @多人（跨部门协作信号）
- 会话结束后 LLM 提取摘要

### 2.2 文档记忆（Document Memory）

**定义**：被引用、讨论或修改的文件的结构化摘要。

**结构**：
```json
{
  "type": "document",
  "id": "doc_20260414_001",
  "title": "AI 推荐决策方案评估",
  "path_or_url": "docs.techcorp.com/doc/0a526f...",
  "accessed_by": ["user_a"],
  "access_time": "2026-04-14T10:00:00+08:00",
  "summary": "CQL 模型 20260309 版本评估，资源节省 4.3%，丢包率 -10.65%",
  "key_facts": ["码率下降 4.3%", "高清覆盖率提升 5.23%"],
  "tags": ["ai-reco", "评估报告", "20260309"]
}
```

### 2.3 人际关系图谱（Relationship Memory）

**定义**：组织内的人员、部门、职责映射，及协作关系。

**结构**：
```json
{
  "type": "relationship",
  "person": "user_b",
  "email": "user_b@company.com",
  "department": "算法工程",
  "role": "算法工程师",
  "expertise": ["强化学习", "推荐系统"],
  "collaboration_history": [
    {"date": "2026-04-10", "topic": "离线评估方案对齐", "outcome": "确认指标体系"}
  ],
  "last_interaction": "2026-04-14",
  "tags": ["算法侧", "RL"]
}
```

**自动触发条件**：
- 检测到新的协作人（@某人、"和某人讨论了"）
- 自动从 org_api 查询组织架构信息补充

### 2.4 事件流记忆（Event Memory）

**定义**：时间线上的里程碑、会议、决策节点。

**结构**：
```json
{
  "type": "event",
  "id": "evt_20260414_001",
  "title": "PriceEngine v2 灰度启动会",
  "time": "2026-04-14T14:00:00+08:00",
  "participants": ["user_a", "user_b", "user_c"],
  "outcome": "批准灰度，比例 1%",
  "linked_decisions": ["conv_20260414_001"],
  "tags": ["灰度", "推荐决策", "里程碑"]
}
```

---

## 三、引擎架构

### 3.1 总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenClaw 智能体层                          │
│    （各部门 AI 实例，通过 Skill 调用 Memory Engine API）        │
└──────────────────────┬──────────────────────────────────────┘
                       │ Skill API（mem_write / mem_search / mem_recall）
┌──────────────────────▼──────────────────────────────────────┐
│                  Memory Engine 中间层                         │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ 写入流水线   │  │  检索引擎    │  │   记忆治理模块      │  │
│  │ AutoExtract │  │ SQLite FTS5  │  │  衰减/压缩/归档     │  │
│  │ ManualWrite │  │ + 可选向量库 │  │  权限过滤           │  │
│  └──────┬──────┘  └──────┬───────┘  └────────────────────┘  │
│         │                │                                     │
│  ┌──────▼────────────────▼──────────────────────────────────┐ │
│  │                  统一存储层                                │ │
│  │                                                           │ │
│  │  memories.db（SQLite）                                    │ │
│  │  ├── conversations（对话记忆）                             │ │
│  │  ├── documents（文档记忆）                                 │ │
│  │  ├── relationships（人际图谱）                             │ │
│  │  └── events（事件流）                                      │ │
│  │                                                           │ │
│  │  embeddings/（可选，ONNX + numpy 向量索引）                  │ │
│  │  archives/（冷存储，超过 90 天自动归档）                    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 写入流水线

**两种写入路径**：

```
Path A（自动）：
会话结束触发 → AutoExtractor（LLM）→ 结构化 JSON → 去重检查 → 写入 DB

Path B（主动）：
用户/智能体显式调用 mem_write → 直接写入 DB
```

**AutoExtractor 提示词模板**（核心）：
```
你是企业记忆提取器。从以下对话中提取结构化记忆：

对话内容：{conversation_text}
参与者：{participants}

请提取：
1. 决策事项（明确的决定/共识）
2. 待办事项（owner + task + due）
3. 提及的文档/资源
4. 新出现的协作人员
5. 关键事件/里程碑

输出格式：JSON，符合 MemoryRecord schema
```

**去重机制**：
- 内容哈希（SHA256）防止完全重复
- 语义相似度阈值（cosine > 0.92）防止语义重复
- 时间窗口（同一话题 1 小时内合并）

### 3.3 检索引擎（双轨制）

**Track A：SQLite FTS5 关键词检索**（零依赖）
```sql
SELECT * FROM memories_fts 
WHERE memories_fts MATCH '上周 灰度决策'
ORDER BY rank LIMIT 10;
```

**Track B：语义向量检索**（可选，零依赖实现）

实际采用 **ONNX Runtime + numpy** 实现，无需安装 ChromaDB/FAISS：

```python
# 用 all-MiniLM-L6-v2 ONNX 模型生成 384 维嵌入
session = ort.InferenceSession("models/all-MiniLM-L6-v2/onnx/model.onnx")
query_emb = encode(session, ["上周跨部门关于灰度比例的决定"])

# numpy 余弦相似度计算（内存索引）
scores = cosine_similarity(query_emb, stored_embeddings)
top_k_ids = np.argsort(scores)[::-1][:top_k]
```

**混合排序**（Hybrid RRF Fusion）：

采用 Reciprocal Rank Fusion（RRF，k=60）而非线性加权，对 rank 位置不敏感：

```
rrf_score(doc) = 1/(k + rank_fts) + 1/(k + rank_vec)
```

等效于近似 `0.4 × BM25_score + 0.6 × semantic_score`（config `hybrid_alpha: 0.6`）。
当向量库不可用时，自动降级为纯 FTS5 模式。

**设计选择说明（SQLite over PostgreSQL，RRF over PPR）**：
- **SQLite**：零运维，Skill 内嵌运行，避免企业用户维护独立数据库；FTS5 + 触发器自动同步，性能足够（检索 P99 < 2ms）
- **ONNX over ChromaDB**：ONNX Runtime 是纯 C++ 运行时，不依赖 Python 向量数据库；模型文件随 Skill 打包，无网络依赖
- **RRF over PPR（Personalized PageRank）**：RRF 无超参、实现简单、在 1K 级记忆库中效果与 PPR 相当；PPR 适合百万级图谱，在本场景下过度设计

### 3.4 记忆治理（生命周期管理）

```
新鲜层（0-7天）  → 完整保留，高频检索
活跃层（7-30天） → 压缩摘要，保留关键决策
沉淀层（30-90天）→ 每月摘要，只保留结论
归档层（90天+）  → 移至 archives/，按需加载
```

**压缩策略**：
- 每周日 00:00 触发 Consolidation Job（可接入 OpenClaw cron）
- 同一参与者同一话题的多条记忆 → 合并为一条综合摘要
- 保留原始记录 ID 作为 provenance 链

### 3.5 多用户权限模型

```
记忆可见性（visibility）：
  public    → 所有人可见（跨部门共享）
  team      → 仅同 department 可见
  private   → 仅 owner 可见

查询过滤：
  mem_search(query, caller_id) 
  → 自动过滤 private（非 owner）
  → 自动过滤 team（非同部门）
  → public 全量可见
```

---

## 四、Skill 接口设计

### 4.1 核心 API

| 接口 | 参数 | 说明 |
|------|------|------|
| `mem_write` | type, content, participants, visibility | 主动写入记忆 |
| `mem_search` | query, type_filter, time_range, caller_id | 语义/关键词检索 |
| `mem_recall` | entity, depth | 查询某人/某项目的完整记忆链 |
| `mem_extract` | conversation_text, participants | 自动提取并写入 |
| `mem_status` | - | 查看记忆库统计 |
| `mem_archive` | before_date | 手动触发归档 |

### 4.2 Skill 触发词（自然语言）

```
写入类：
- "记录一下刚才的决策"
- "把这次讨论存到记忆库"
- "记住：xxx负责xxx"

检索类：
- "找一下上周和产品侧的决策"
- "xxx上次说的方案是什么"
- "3月份关于灰度比例的约定"

人际类：
- "xxx是负责什么的"
- "和推荐组的协作记录"

状态类：
- "记忆库里有多少条记录"
- "最近一个月存了什么"
```

---

## 五、安装与集成

### 5.1 零依赖模式（推荐初始安装）

```bash
# 1. 将 Skill 复制到 workspace
cp -r enterprise-memory ~/.openclaw/workspace/skills/

# 2. 初始化记忆库
python3 ~/.openclaw/workspace/skills/enterprise-memory/scripts/init_db.py

# 3. 在 AGENTS.md 中注册 Skill（可选，用于心跳自动触发）
echo "- enterprise-memory: 企业记忆引擎，跨部门协作记忆管理" >> ~/.openclaw/workspace/AGENTS.md
```

### 5.2 向量检索扩展（可选，零额外依赖）

模型已随 Skill 打包（`models/all-MiniLM-L6-v2/onnx/model.onnx`），只需安装 ONNX Runtime：

```bash
pip install onnxruntime numpy
# 重新初始化，自动检测并启用向量模式
python3 scripts/init_db.py --with-vectors
```

> **无需** ChromaDB、FAISS 或任何向量数据库。嵌入向量以 numpy `.npy` 格式存储在 `embeddings/` 目录，内存加载后使用余弦相似度检索。

### 5.3 配置文件

```yaml
# skills/enterprise-memory/config.yaml
storage:
  backend: sqlite          # sqlite（默认）| sqlite+onnx（启用向量检索）
  db_path: ~/.openclaw/workspace/memory_engine/memories.db
  embeddings_path: ~/.openclaw/workspace/memory_engine/embeddings/

retention:
  fresh_days: 7
  active_days: 30
  archive_days: 90

extraction:
  auto_extract: true       # 每次跨部门会话后自动触发
  min_participants: 2      # 触发自动提取的最少参与人数
  llm_model: default       # 使用 OpenClaw 默认模型

multi_user:
  enabled: true
  default_visibility: team

search:
  hybrid_alpha: 0.6        # RRF 融合中语义侧权重比例（近似，实际采用 RRF 排序）
  top_k: 10
  time_decay: true         # 新记忆权重更高
```

---

## 六、与 context-slim 的关系

本引擎与 `context-slim` Skill 形成互补，而非替代：

| 维度 | context-slim | enterprise-memory |
|------|------|------|
| 目标 | 减少上下文 token 消耗 | 增强长程记忆能力 |
| 方向 | 精简（删） | 扩充（存） |
| 用户 | 单人 | 多人共享 |
| 触发 | 手动 | 自动 + 手动 |
| 存储 | 文件系统 | SQLite + 可选向量库 |

**推荐组合使用**：
- `enterprise-memory` 存入重要记忆
- `context-slim` 定期清理 MEMORY.md，防止上下文膨胀
- `memoryres/` 作为冷存储中间层

---

## 七、评测维度设计与实测结果

（完整方法论见 `docs/evaluation.md`）

### 7.1 企业场景专项评测（E1-E7）

基于 6 场中文企业会议（星辰电商 PriceEngine v2 智能定价系统灰度发布场景）手工标注，12 条检索 QA，10 条权限用例：

| 维度 | 指标 | 目标 | **实测** |
|------|------|------|---------|
| E1 自动提取准确率 | 决策 F1 / 写入成功率 | ≥ 0.85 / 100% | **1.000 / 100%** ✅ |
| E2 跨人协作检索召回 | R@3 / 通过率 | ≥ 0.60 / 60% | **0.857 / 100%** ✅ |
| E3 知识图谱命中率 | 人员召回 / 决策命中率 | ≥ 0.50 / 60% | **0.667 / 67%** ✅ |
| E4 长程时序检索 | 通过率 | ≥ 75% | **100%** ✅ |
| E5 权限隔离准确率 | 隔离准确率 | 100% | **100%** ✅ |
| E6 去重效果 | 准确率 | ≥ 0.95 | **100%** ✅ |
| E7 性能基准 | 检索 P99 / 写入 P99 | ≤ 500ms / ≤ 100ms | **0.7ms / 34ms** ✅ |

**7/7 维度全部通过**，总耗时 ≈ 2s。

### 7.2 LoCoMo-10 通用检索召回评测

基于 LoCoMo 个人对话数据集，1536 个有效 QA（跳过 Cat5 对抗题）：

| 方法 | R@1 | R@5 | MRR |
|------|-----|-----|-----|
| FTS（BM25 纯关键词） | 0.238 | 0.433 | 0.345 |
| 向量（MiniLM-L6-v2 ONNX） | 0.110 | 0.334 | 0.223 |
| **Hybrid RRF（最终）** | **0.289** | **0.549** | **0.431** |

Hybrid 相比纯 FTS：R@5 **+11.6pp**，MRR **+8.5pp**。

---

*本白皮书对应实现代码见 `scripts/` 目录，评测报告见 `docs/evaluation.md`，Demo 脚本见 `demo/` 目录。*
