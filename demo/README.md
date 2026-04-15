# demo/ — 端到端 Demo 与评测结果

## 文件说明

| 文件 | 说明 |
|------|------|
| `demo_enterprise.py` | 端到端 Demo 脚本，演示 5 个跨部门协作场景 |
| `eval_enterprise.py` | 企业场景专项评测脚本（E1-E7，可独立运行） |
| `eval_enterprise_report.json` | **企业场景专项评测报告（7/7 全部通过）** |
| `eval_report_final.json` | LoCoMo-10 检索召回评测（1536 QA，三路对比） |
| `full_test_report.json` | 55 个单元测试报告（全部通过） |
| `scale_test_report.json` | 1336 个规模压力测试报告（全部通过） |

---

## 运行 Demo

```bash
# 进入项目根目录
cd enterprise-memory/

# 快速运行（临时 DB，结束自动清理）
python3 demo/demo_enterprise.py

# 展示完整决策 & 待办详情
python3 demo/demo_enterprise.py --verbose

# 保留 DB，后续可继续 CLI 查询
python3 demo/demo_enterprise.py --keep-db
```

## 运行企业专项评测

```bash
# 全量跑（E1-E7）
python3 demo/eval_enterprise.py

# 单独跑某一维度
python3 demo/eval_enterprise.py --section E1
python3 demo/eval_enterprise.py --section E2 --verbose
```

---

## Demo 场景一览

> 场景背景：**星辰电商 PriceEngine v2 智能定价系统**从灰度到全量发布的完整协作链路  
> 角色：李明(alice，算法工程师) × 王芳(bob，产品经理) × 张伟(carol，实习生) × 刘强(david，基础设施) × 赵雪(eve，数据洞察) × 陈总(frank，产品总监) × 孙磊(grace，算法负责人)

| 场景 | 角色 | 验证能力 |
|------|------|---------|
| 场景一：PriceEngine v2 灰度方案评审会 | 李明(alice，算法) × 王芳(bob，产品运营) | 自动提取决策/待办、四类记忆写入 |
| 场景二：新人张伟查询历史灰度决策 | 张伟(carol，算法工程实习生) | 跨人 FTS 检索、team 权限可见性 |
| 场景三：知识图谱关联扩展 | 张伟(carol，算法工程) | BFS 图谱扩展、实体-关系三元组 |
| 场景四：李明查与王芳的协作链路 | 李明(alice) 查 王芳(bob) 的协作记录 | Recall 记忆链，跨部门关系溯源 |
| 场景五：长程跨会话持久化 | 张伟(carol) 在新会话找历史决策 | 跨会话持久化、权限隔离验证 |

---

## 评测结果摘要

### 企业场景专项评测（`eval_enterprise_report.json`）

> 场景：星辰电商 PriceEngine v2 智能定价系统灰度发布 | 6 场会议 × 7 人 × 4 部门  
> 12 条检索 QA，10 条权限用例，总耗时 ≈ 2s

| 维度 | 核心指标 | 结果 |
|------|---------|------|
| E1 自动提取准确率 | 决策 F1 / 写入成功率 | **1.000 / 100%** |
| E2 跨人协作检索召回 | R@3 / 通过率 | **0.857 / 100%** |
| E3 知识图谱命中率 | 人员召回 / 决策命中率 | **0.667 / 67%** |
| E4 长程时序检索 | 通过率 | **100%** |
| E5 权限隔离准确率 | 隔离准确率 | **100%** |
| E6 去重效果 | 准确率 | **100%** |
| E7 性能基准 | 检索 P99 / 写入 P99 | **0.7ms / 34ms** ✓ |

**7/7 维度全部通过 ✅**

---

### LoCoMo-10 检索召回（`eval_report_final.json`）

> 数据集：10 个多轮对话样本，1536 个有效 QA（跳过 450 个 Cat5 对抗题）

| 方法 | R@1 | R@3 | R@5 | MRR |
|------|-----|-----|-----|-----|
| FTS（BM25） | 0.238 | 0.371 | 0.433 | 0.345 |
| 向量（MiniLM ONNX） | 0.110 | 0.263 | 0.334 | 0.223 |
| **Hybrid（最终）** | **0.289** | **0.457** | **0.549** | **0.431** |

Hybrid 相比纯 FTS：R@5 **+11.6pp**，MRR **+8.5pp**。

### 单元测试（`full_test_report.json`）

- 55 个用例 / **100% pass rate**
- 覆盖：写入、去重、检索、权限、自动提取、治理

### 规模压力测试（`scale_test_report.json`）

- 1336 个用例 / **100% pass rate**
- 10 大压测板块：写入压力(200)、去重矩阵(100)、FTS 参数化(500)、权限矩阵(36)、提取容错(100)、治理压力(50)、Embedder(100)、并发安全(50)、数据边界(100)、召回回归(100)
