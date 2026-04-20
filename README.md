# Enterprise Memory — 企业级长程协作记忆引擎


为 AI 智能体提供跨部门、跨周级的持久化记忆管理能力。


## 核心特性


- **四类记忆**：对话 / 文档 / 人际 / 事件
- **混合检索**：语义向量 + BM25 融合，知识图谱多跳查询
- **多用户权限**：public / team / private 三级隔离
- **自动提取**：跨部门会话后自动提取决策与待办
- **记忆治理**：自动压缩、归档、过期清理
- **零依赖**：仅需 Python + SQLite，可选向量扩展


## 快速开始


```bash
python3 scripts/init_db.py          # 初始化数据库
python3 scripts/memory_engine.py status  # 验证
```


向量检索扩展（可选）：


```bash
pip install onnxruntime numpy
python3 scripts/init_db.py --with-vectors
```


## 使用


```bash
# 写入
python3 scripts/memory_engine.py write --type conversation --content '{"summary":"..."}' --owner alice


# 检索
python3 scripts/memory_engine.py search --query "上周灰度决策" --caller alice


# 自动提取
python3 scripts/memory_engine.py extract --session-file session.json --participants alice,bob


# 记忆治理
python3 scripts/memory_engine.py consolidate
```


## 项目结构


```
├── config.yaml              # 配置
├── scripts/
│   ├── memory_engine.py     # 核心引擎
│   ├── embedder.py          # 向量嵌入
│   ├── graph_extractor.py   # 知识图谱提取
│   └── init_db.py           # 数据库初始化
├── docs/
│   ├── whitepaper.md        # 架构白皮书
│   └── evaluation.md        # 评测报告
├── models/                  # 本地嵌入模型 (ONNX)
├── tests/                   # 测试与评测
└── demo/                    # 演示脚本
```


## 文档


- [架构白皮书](docs/whitepaper.md)
- [评测报告](docs/evaluation.md)
- [Skill 说明](SKILL.md)


## License


MIT · **liufuyao** · 2026
