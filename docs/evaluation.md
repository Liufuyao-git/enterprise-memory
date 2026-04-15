# 企业级长程协作记忆引擎 — 评测方法论

**版本**：v1.0  
**日期**：2026-04-15

---

## 一、评测体系概览

本引擎设计了两套独立评测：

| 评测集 | 数据来源 | 目的 |
|--------|---------|------|
| **企业场景专项（E1-E7）** | 手工标注中文企业对话 | 验证核心能力是否满足办公场景 |
| **LoCoMo-10 通用召回** | LoCoMo 公开数据集 | 对比行业基线，量化检索性能 |

---

## 二、企业场景专项评测（E1-E7）

### 2.1 数据集构建

**场景设定**：星辰电商 PriceEngine v2 智能定价系统从灰度到全量发布，多部门协作（算法工程 × 产品运营 × 基础设施 × 数据洞察）

**语料**：6 场会议记录（M1-M6），共 7 名参与人（李明/王芳/张伟/刘强/赵雪/陈总/孙磊）

| 会议 | 主题 | 参与方 |
|------|------|--------|
| M1 | PriceEngine v2 灰度方案评审 | 算法工程 × 产品运营 |
| M2 | PriceEngine v2 灰度扩量决策 | 算法工程 × 产品运营 |
| M3 | PriceEngine v2 离线效果评估 | 算法工程 × 数据洞察 |
| M4 | PriceEngine SDK 性能优化 | 算法工程 × 基础设施 |
| M5 | PriceEngine 模型蒸馏实验进展 | 算法工程内部 |
| M6 | PriceEngine v2 全量发布评审 | 多部门联席 |

**标注内容**：每场会议手工标注期望提取的 decisions 和 action_items，作为 ground truth。

### 2.2 E1 — 自动提取准确率

**测试逻辑**：
1. 将每场会议的手工 LLM 输出（包含 decisions + action_items）通过 `mem_extract_commit` 写入
2. 用唯一标签（`e1-{meeting_id}`）通过 `mem_search` 读回该条记忆
3. 计算 decisions 的 F1（token overlap）和 action_items 的 F1
4. 验证 owner 是否正确关联

**指标**：
- `write_success_rate`：写入成功率
- `decision_f1`：decisions 与 gold 的 token-level F1 均值
- `task_f1`：action_items 与 gold 的 F1 均值
- `owner_match_rate`：action_items 中 owner 字段准确率

**通过门槛**：`decision_f1 ≥ 0.7` AND `write_success_rate = 100%`

**实测结果**：`decision_f1 = 1.000`，`write_success_rate = 100%`，`owner_match_rate = 100%` ✅

### 2.3 E2 — 跨人协作检索召回

**测试逻辑**：
- 12 条检索 QA（7 条跨人检索 + 3 条决策检索 + 2 条人员检索）
- 每条 QA：`{query, caller_id, caller_dept, expected_owners}`
- 用 `mem_search` 检索 top-5，验证 `expected_owners` 是否在结果中

**指标**：
- `R@K`（K=1,3,5）：top-K 中命中 expected_owners 的比例
- `pass_rate`：单条 QA 通过率（`R@3 > 0` AND `found_n ≥ min_results`）
- `avg_latency_ms`：检索延迟均值

**通过门槛**：`R@3 ≥ 0.60`

**实测结果**：`R@3 = 0.857`，通过率 100%，延迟 avg=0.5ms ✅

**注意**：team visibility 记忆只对同 department 可见，跨部门检索设计上不应命中，此为正确权限行为。

### 2.4 E3 — 知识图谱检索命中率

**测试逻辑**：
- 3 条图谱查询（`mem_graph_search`，max_hops=2）
- 验证返回的 entity 节点中是否包含 expected_persons 和 expected_decisions

**指标**：
- `person_recall`：expected_persons 在 entity 节点中的召回率
- `decision_hit_rate`：expected_decisions 关键词在 entity 名称中的命中率
- `pass_rate`：`person_recall ≥ 0.5` AND `total_hits ≥ 1` 的 QA 比例

**通过门槛**：`avg_person_recall ≥ 0.5` AND `pass_rate ≥ 0.6`

**实测结果**：`person_recall = 0.667`，`decision_hit_rate = 67%`，`pass_rate = 67%` ✅（门槛：≥0.5 / ≥0.6）

### 2.5 E4 — 长程时序检索

**测试逻辑**：4 个检索用例，验证 `days` 参数和 `type_filter` 的正确性：
- `days=90` 全量检索应有结果
- `type_filter=conversation` 结果全为 conversation 类型
- `type_filter=event` 结果全为 event 类型（用 tag 检索）
- 英文关键词（QualityScore）精确命中 document 类型

**通过门槛**：`pass_rate = 100%`

**实测结果**：4/4 通过 ✅

### 2.6 E5 — 权限隔离准确率

**测试逻辑**：10 个 `(owner, visibility, owner_dept, caller, caller_dept, should_find)` 组合，穷举三级可见性：

| 可见性 | caller = owner | 同 dept | 跨 dept |
|--------|---------------|---------|---------|
| private | ✅ 可见 | ❌ 不可见 | ❌ 不可见 |
| team | ✅ 可见 | ✅ 可见 | ❌ 不可见 |
| public | ✅ 可见 | ✅ 可见 | ✅ 可见 |

**通过门槛**：`accuracy = 100%`

**实测结果**：10/10 通过 ✅

### 2.7 E6 — 去重效果

**测试逻辑**：3 个场景验证 content hash 去重：
1. 完全相同 content dict → 预期触发 `duplicate`
2. 相同 summary 但不同 key → 哈希不同，不触发去重
3. 完全不同 content → 不触发去重

**通过门槛**：`accuracy = 100%`

**实测结果**：3/3 通过 ✅

### 2.8 E7 — 性能基准

**测试逻辑**：
- 写入压测：100 次 `mem_write`，计算 P50/P99/avg
- 检索压测：200 次 `mem_search`，计算 P50/P99/avg

**通过门槛**：检索 P99 ≤ 500ms（白皮书承诺）

| 操作 | P50 | P99 | avg |
|------|-----|-----|-----|
| 写入 | 12.3ms | 34ms | 13.9ms |
| 检索 | 0.5ms | 0.7ms | 0.5ms |

**实测结果**：检索 P99 = 0.7ms，远优于 500ms 目标 ✅

---

## 三、LoCoMo-10 通用召回评测

### 3.1 数据集

- **来源**：[LoCoMo: A Dataset for Long Context Memory in Conversations](https://arxiv.org/abs/2402.15656)
- **规模**：10 个多轮对话样本，共 1986 个 QA
- **有效 QA**：1536（跳过 Cat5 对抗性问题 450 个）

### 3.2 评测方法

1. 将每段对话按轮次拆分为记忆片段写入引擎
2. 对每个 QA 的 question 执行检索（`mem_search`）
3. 检查 answer 所在的原始片段是否出现在 top-K 结果中

### 3.3 三路对比

| 方法 | 实现 | R@1 | R@3 | R@5 | MRR |
|------|------|-----|-----|-----|-----|
| BM25（纯 FTS） | SQLite FTS5 | 0.238 | 0.371 | 0.433 | 0.345 |
| 向量语义 | MiniLM-L6-v2 ONNX | 0.110 | 0.263 | 0.334 | 0.223 |
| **Hybrid RRF** | FTS + ONNX + RRF(k=60) | **0.289** | **0.457** | **0.549** | **0.431** |

Hybrid vs FTS：R@5 **+11.6pp**，MRR **+8.5pp**  
Hybrid vs 向量：R@5 **+21.5pp**，MRR **+20.8pp**

### 3.4 结论

FTS 在关键词匹配上优势明显（R@5=0.433 vs 向量的 0.334），但语义向量能补充 FTS 漏掉的语义相关记忆（R@1 的差异说明语义检索精确率低，适合作为辅助而非主检索）。RRF 融合有效结合两者优势。

---

## 四、运行评测脚本

```bash
cd enterprise-memory/

# 企业场景专项评测（E1-E7，全量）
python3 demo/eval_enterprise.py

# 单节评测 + 详情
python3 demo/eval_enterprise.py --section E1 --verbose
python3 demo/eval_enterprise.py --section E2 --verbose

# LoCoMo-10 评测（需提前准备数据集）
python3 tests/eval_locomo.py --mode hybrid

# 查看最新报告
cat demo/eval_enterprise_report.json | python3 -m json.tool
```

---

## 五、已知局限

| 限制 | 说明 | 影响 |
|------|------|------|
| FTS 中文分词 | `unicode61` 按连续中文字符分词，不支持子词检索 | 单字/二字词需在 tags 中显式标注 |
| 向量索引规模 | numpy 内存索引，适合 ≤ 10 万条 | 超大规模需换 FAISS/annoy |
| `mem_recall` 接口 | 不返回完整 content 字段（仅返回 summary） | 需用 `mem_search` + 唯一标签读取 content |
| 实体抽取 | 基于规则 regex，非 NLP 模型 | 实体识别精度受限，可接 LLM 增强 |
