#!/usr/bin/env python3
"""
eval_locomo.py — LoCoMo-10 向量检索召回率评测

数据集结构：
  - 10个样本，每个有多个session的对话（D1/D2/D3...）
  - 每条对话 turn 有 dia_id（如 "D1:3"）
  - 每个QA有 evidence（dia_id列表），ground truth 就是这些 turn

评测逻辑：
  1. 将所有对话 turn 编码为 embedding（index）
  2. 用问题 embedding 在 index 中检索 top-K
  3. 看 ground truth evidence 是否出现在 top-K 里
  4. 指标：Recall@1, Recall@3, Recall@5, MRR

运行：
  python3 scripts/eval_locomo.py                        # 全量评测
  python3 scripts/eval_locomo.py --top-k 5              # 只看 top-5
  python3 scripts/eval_locomo.py --method fts           # 只跑 FTS 基线
  python3 scripts/eval_locomo.py --method vector        # 只跑向量检索
  python3 scripts/eval_locomo.py --method both          # 两种都跑（默认）
  python3 scripts/eval_locomo.py --category 1           # 只评测某类别
  python3 scripts/eval_locomo.py --sample-id 0          # 只评测第0个样本
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 路径设置
_SCRIPT_DIR = Path(__file__).parent
_REPO_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

DATA_PATH = _REPO_DIR / "data" / "locomo10.json"

# ── category 说明 ──────────────────────────────────────────────
CATEGORY_NAMES = {
    1: "单跳事实 (Single-hop Fact)",
    2: "时间推理 (Temporal)",
    3: "多跳推理 (Multi-hop)",
    4: "开放域 (Open-ended)",
    5: "对抗/无法回答 (Adversarial/Unanswerable)",
}


# ══════════════════════════════════════════════════════════════
# 数据加载与解析
# ══════════════════════════════════════════════════════════════

def load_dataset(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_dia_id(dia_id: str) -> Tuple[int, int]:
    """'D2:5' → (2, 5)"""
    m = re.match(r"D(\d+):(\d+)", dia_id)
    if not m:
        raise ValueError(f"无法解析 dia_id: {dia_id!r}")
    return int(m.group(1)), int(m.group(2))


def extract_turns(
    sample: dict,
    speaker_prefix: bool = True,
    date_prefix: bool = True,
) -> Dict[str, str]:
    """
    提取单个样本的所有对话 turn。
    返回：{dia_id: enriched_text}

    enriched_text 格式（默认）：
      "[2023-05-08] Caroline: I went to a LGBTQ support group yesterday."

    speaker_prefix: 把说话人名字拼到 turn 文本前面
      → 解决 Cat1 第三人称/第一人称错配问题
    date_prefix: 把 session 日期拼到最前面
      → 解决 Cat2 时间类问题（"yesterday" 需要知道 session 日期）
    """
    conv = sample["conversation"]
    turns: Dict[str, str] = {}

    # 预提取 session 日期（session_N_date_time 字段）
    session_dates: Dict[int, str] = {}
    if date_prefix:
        for k, v in conv.items():
            m = re.match(r"session_(\d+)_date_time", k)
            if m and isinstance(v, str):
                idx = int(m.group(1))
                # 提取日期部分：'1:56 pm on 8 May, 2023' → '8 May 2023'
                date_m = re.search(r"(\d{1,2}\s+\w+,?\s+\d{4})", v)
                session_dates[idx] = date_m.group(1).replace(",", "") if date_m else v

    session_keys = sorted(
        [k for k in conv if re.match(r"session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )
    for sk in session_keys:
        session_idx = int(sk.split("_")[1])
        date_str = session_dates.get(session_idx, "")
        for turn in conv[sk]:
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "").strip()
            speaker = turn.get("speaker", "").strip()
            if not dia_id or not text:
                continue

            parts = []
            if date_prefix and date_str:
                parts.append(f"[{date_str}]")
            if speaker_prefix and speaker:
                parts.append(f"{speaker}:")
            parts.append(text)
            turns[dia_id] = " ".join(parts)

    return turns


# 停用词：英文疑问句头词 + 冗余功能词
_QUERY_STOPWORDS = {
    "what", "where", "when", "who", "why", "how", "which", "whose",
    "did", "does", "do", "has", "have", "had", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "at", "on", "to", "for", "with",
    "that", "this", "it", "he", "she", "they", "we", "you", "i",
    "and", "or", "but", "not",
}


def preprocess_query(query: str) -> str:
    """
    清洗 query：去掉疑问词/停用词，保留名词短语和实体。

    'What did Caroline research?' → 'Caroline research'
    'Where has Melanie camped?'   → 'Melanie camped'
    'When did Caroline go to the LGBTQ support group?' → 'Caroline LGBTQ support group'
    """
    tokens = re.sub(r"[?!.,]", "", query).split()
    kept = [t for t in tokens if t.lower() not in _QUERY_STOPWORDS]
    # 至少保留 2 个词，否则回退到原始 query
    return " ".join(kept) if len(kept) >= 2 else query


# ══════════════════════════════════════════════════════════════
# HyDE（Hypothetical Document Expansion）
# ══════════════════════════════════════════════════════════════

# 属性词 → 同义词/相关词（用于 BM25 扩展，解决 Cat1 词汇鸿沟）
_ATTR_SYNONYMS: Dict[str, List[str]] = {
    "identity":            ["transgender", "trans", "LGBTQ", "gender", "sexuality"],
    "relationship status": ["single", "married", "dating", "partner", "boyfriend", "girlfriend"],
    "career":              ["job", "work", "career", "counseling", "mental health", "working"],
    "career path":         ["career", "counseling", "working", "profession", "mental health"],
    "hobby":               ["hobby", "activities", "interest", "pottery", "running", "painting"],
    "hobbies":             ["hobby", "activities", "interest", "pottery", "running"],
    "job":                 ["work", "job", "career", "profession", "working"],
    "education":           ["school", "study", "university", "degree", "learning"],
    "book":                ["read", "book", "novel", "author", "reading"],
    "books":               ["read", "book", "novel", "reading", "author"],
    "activities":          ["activities", "did", "joined", "participated", "hobby"],
    "activity":            ["activities", "joined", "participated"],
    "destress":            ["de-stress", "relax", "running", "headspace", "pottery"],
    "political":           ["liberal", "conservative", "vote", "political"],
    "politics":            ["liberal", "conservative", "vote", "political"],
    "opinion":             ["think", "feel", "believe", "opinion"],
    "pets":                ["dog", "cat", "pet", "animal"],
    "family":              ["family", "kids", "children", "parent", "spouse"],
    "health":              ["health", "medical", "sick", "hospital", "doctor"],
    "mental health":       ["therapy", "counseling", "mental", "anxiety"],
    "event":               ["event", "joined", "participated", "attended", "mentorship"],
    "events":              ["event", "joined", "participated", "attended"],
    "dream":               ["dream", "hope", "aspire", "goal", "plan"],
    "goal":                ["goal", "plan", "aspire", "dream"],
    "volunteer":           ["volunteer", "help", "community", "mentorship"],
    "language":            ["language", "speak", "learn", "fluent"],
    "art":                 ["art", "paint", "draw", "pottery"],
    "painting":            ["paint", "painted", "artwork", "canvas"],
    "camping":             ["camp", "camping", "outdoor", "hike", "nature"],
    "museum":              ["museum", "exhibit", "gallery"],
    "adoption":            ["adopt", "adoption", "family", "child", "foster"],
    "research":            ["research", "study", "investigate", "look into"],
    "friends":             ["friend", "friendship", "social", "close"],
    "sport":               ["sport", "running", "exercise", "gym", "workout"],
    "sports":              ["sport", "running", "exercise", "gym"],
    "challenge":           ["challenge", "difficult", "struggle", "overcome"],
    "achievement":         ["achievement", "accomplish", "proud", "success"],
    "social":              ["social", "friend", "community", "people"],
}

_ATTRIBUTE_PATTERNS = [
    re.compile(r"what (?:is|are|was|were) [\w]+'s", re.I),
    re.compile(r"what (?:does|do|did) [\w]+ (?:do|like|enjoy|prefer)", re.I),
]


def hyde_fts_expand(question: str) -> str:
    """
    HyDE for BM25：对属性型 query 追加同义词扩展，其他 query 不变。

    "What is Caroline's identity?"     → 原词 + transgender trans LGBTQ gender
    "What career path has Caroline...?" → 原词 + career counseling working profession
    "When did Melanie paint a sunrise?" → 原词（时间型不扩展）
    """
    ql = question.lower()
    extra: List[str] = []

    # 只对属性型 query 扩展
    is_attr = any(p.match(ql) for p in _ATTRIBUTE_PATTERNS)
    if not is_attr:
        # 宽松匹配：含 "what is/are X's Y" 形式（包含所有格）
        is_attr = bool(re.search(r"[\w]+'s\s+\w", ql))

    if is_attr:
        for key, syns in _ATTR_SYNONYMS.items():
            if key in ql:
                extra.extend(syns[:4])  # 每个 key 最多取 4 个同义词

    base = preprocess_query(question)
    if extra:
        deduped = list(dict.fromkeys(extra))[:10]  # 最多 10 个额外词
        return base + " " + " ".join(deduped)
    return base


def hyde_vector_queries(question: str) -> List[str]:
    """
    HyDE for Vector：生成假设性陈述句，用于向量检索。
    返回 [原始query] + [0-2个假设句]，调用方取均值 embedding。

    "What did Caroline research?"   → ["Caroline research", "Caroline was researching"]
    "What is Caroline's identity?"  → ["Caroline identity", "Caroline is transgender"]
    """
    q = question.strip().rstrip("?")
    ql = q.lower()
    hyps: List[str] = [preprocess_query(question)]

    # 模式1: What did/does/do/has/have X [pred]
    m = re.match(r"what (?:did|does|do|has|have) ([\w']+)\s+(.+)", ql)
    if m:
        subj = m.group(1).rstrip("'s").title()
        pred = m.group(2)
        hyps.append(f"{subj} {pred}")
        hyps.append(f"I {pred}")

    # 模式2: What is X's [attr]
    m = re.match(r"what is ([\w]+)'s\s+(.+)", ql)
    if m:
        subj = m.group(1).title()
        attr = m.group(2)
        hyps.append(f"{subj} {attr}")
        hyps.append(f"I am {attr}")

    # 模式3: Where/When X [pred]
    m = re.match(r"(?:where|when) (?:has|have|did|does|is) ([\w]+)\s*(.+)?", ql)
    if m:
        subj = m.group(1).title()
        pred = (m.group(2) or "").strip()
        hyps.append(f"{subj} {pred}".strip())
        hyps.append(f"I {pred}".strip())

    # 模式4: How long/many X [pred]
    m = re.match(r"how (?:long|many|much|often|far) (?:has|have|did|does) ([\w]+)\s+(.+)", ql)
    if m:
        subj = m.group(1).title()
        pred = m.group(2)
        hyps.append(f"{subj} {pred}")
        hyps.append(f"I {pred}")

    return list(dict.fromkeys(h for h in hyps if len(h.strip()) > 3))[:5]


# ══════════════════════════════════════════════════════════════
# 评测指标
# ══════════════════════════════════════════════════════════════

def recall_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> float:
    """
    Recall@K：top-K 里有多少 ground truth 被召回。
    返回 [0, 1]，多 evidence 时取平均。
    """
    if not ground_truth:
        return 0.0
    top_k_set = set(retrieved[:k])
    hits = sum(1 for ev in ground_truth if ev in top_k_set)
    return hits / len(ground_truth)


def reciprocal_rank(retrieved: List[str], ground_truth: List[str]) -> float:
    """MRR：第一个命中 ground truth 的排名的倒数"""
    gt_set = set(ground_truth)
    for rank, item in enumerate(retrieved, start=1):
        if item in gt_set:
            return 1.0 / rank
    return 0.0


# ══════════════════════════════════════════════════════════════
# BM25 检索（替换朴素 TF 词频）
# ══════════════════════════════════════════════════════════════

class BM25Index:
    """
    轻量 BM25 索引，零外部依赖。

    改进点（vs 原 TF 词频）：
      1. IDF 加权：罕见词权重更高，停用词权重趋近 0
      2. TF 饱和：重复词不线性累加（k1 参数控制）
      3. 文档长度归一化：长 turn 不再天然占优（b 参数控制）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._dia_ids: List[str] = []
        self._tf: List[Dict[str, int]] = []    # per-doc term freq
        self._df: Dict[str, int] = {}          # document frequency
        self._dl: List[int] = []               # document lengths
        self._avgdl: float = 0.0
        self._n: int = 0

    # ── 文本预处理 ────────────────────────────────────────────
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """小写 + 按空白切词（与 query 预处理保持一致）"""
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    # ── 建索引 ────────────────────────────────────────────────
    def build(self, turns: Dict[str, str]) -> "BM25Index":
        self._dia_ids = list(turns.keys())
        self._n = len(self._dia_ids)
        self._tf = []
        self._df = {}
        self._dl = []

        for did in self._dia_ids:
            tokens = self.tokenize(turns[did])
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            self._dl.append(len(tokens))
            for t in set(tf):
                self._df[t] = self._df.get(t, 0) + 1

        self._avgdl = sum(self._dl) / self._n if self._n else 1.0
        return self

    # ── 查询 ─────────────────────────────────────────────────
    def scores(self, query: str) -> Dict[str, float]:
        import math
        q_terms = self.tokenize(query)
        result: Dict[str, float] = {}

        for i, did in enumerate(self._dia_ids):
            score = 0.0
            dl_norm = 1 - self.b + self.b * self._dl[i] / self._avgdl
            for term in q_terms:
                tf = self._tf[i].get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * dl_norm)
                score += idf * tf_norm
            result[did] = score

        return result

    def search(self, query: str, top_k: int) -> List[str]:
        sc = self.scores(query)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        return [did for did, _ in ranked[:top_k]]


# BM25 索引按样本缓存（同一样本多次查询只建一次）
_bm25_cache: Dict[str, BM25Index] = {}


def _get_bm25(turns: Dict[str, str], cache_key: str = "") -> BM25Index:
    """获取（或新建）BM25 索引，turns 不变时直接复用"""
    key = cache_key or id(turns)
    if key not in _bm25_cache:
        _bm25_cache[key] = BM25Index().build(turns)
    return _bm25_cache[key]


def fts_scores(
    query: str,
    turns: Dict[str, str],
    cache_key: str = "",
) -> Dict[str, float]:
    """BM25 得分，接口兼容原 TF 词频版本"""
    return _get_bm25(turns, cache_key).scores(query)


def fts_search(
    query: str,
    turns: Dict[str, str],
    top_k: int,
    cache_key: str = "",
) -> List[str]:
    """BM25 检索，接口兼容原 TF 词频版本"""
    return _get_bm25(turns, cache_key).search(query, top_k)


# ══════════════════════════════════════════════════════════════
# 向量检索
# ══════════════════════════════════════════════════════════════

class VectorIndex:
    """
    基于 numpy 的内存向量索引（余弦相似度），支持磁盘缓存。
    适合 LoCoMo-10 规模（每个样本约 50~700 个 turn）。

    缓存文件：
      <cache_dir>/<cache_key>.npy      — 向量矩阵 (N, 384)
      <cache_dir>/<cache_key>.ids.json — dia_id 列表
    """

    def __init__(self, embedder, cache_dir: Optional[Path] = None):
        self.embedder = embedder
        self.dia_ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None  # (N, 384)
        self.cache_dir = cache_dir

    def _cache_paths(self, cache_key: str):
        mat_path = self.cache_dir / f"{cache_key}.npy"
        ids_path = self.cache_dir / f"{cache_key}.ids.json"
        return mat_path, ids_path

    def _try_load_cache(self, cache_key: str) -> bool:
        """尝试从磁盘加载缓存，成功返回 True"""
        if not self.cache_dir or not cache_key:
            return False
        mat_path, ids_path = self._cache_paths(cache_key)
        if mat_path.exists() and ids_path.exists():
            self.matrix = np.load(str(mat_path))
            with open(ids_path, encoding="utf-8") as f:
                self.dia_ids = json.load(f)
            return True
        return False

    def _save_cache(self, cache_key: str):
        """将当前索引写入磁盘"""
        if not self.cache_dir or not cache_key:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        mat_path, ids_path = self._cache_paths(cache_key)
        np.save(str(mat_path), self.matrix)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.dia_ids, f, ensure_ascii=False)

    def build(
        self,
        turns: Dict[str, str],
        cache_key: str = "",
        context_window: int = 0,
    ) -> bool:
        """
        将所有 turn 编码并建立索引。
        如果 cache_key 非空且缓存存在，直接加载（返回 True 表示命中缓存）。

        context_window: 编码时拼接的前后 turn 数量。
          0 = 仅当前 turn（原始行为）
          1 = 前1 + 当前 + 后1，共最多3条拼接
          2 = 前2 + 当前 + 后2，共最多5条拼接
        """
        # 尝试加载缓存（cache_key 含 window 信息，不同 window 分开存）
        full_key = f"{cache_key}_w{context_window}" if cache_key else ""
        if full_key and self._try_load_cache(full_key):
            return True  # cache hit

        self.dia_ids = list(turns.keys())

        if context_window == 0:
            # 原始行为：直接编码单条
            texts = [turns[d] for d in self.dia_ids]
        else:
            # 上下文扩展：把前后 window 条 turn 的文本拼进来
            id_list = self.dia_ids  # 保序
            id_to_idx = {did: i for i, did in enumerate(id_list)}
            texts = []
            for i, did in enumerate(id_list):
                lo = max(0, i - context_window)
                hi = min(len(id_list) - 1, i + context_window)
                ctx_texts = [turns[id_list[j]] for j in range(lo, hi + 1)]
                # 中间那条加 [SEP] 分隔，突出当前 turn
                center_offset = i - lo
                ctx_texts[center_offset] = "[SEP] " + ctx_texts[center_offset] + " [SEP]"
                texts.append(" ".join(ctx_texts))

        self.matrix = self.embedder.encode(texts, normalize=True)

        if full_key:
            self._save_cache(full_key)

        return False  # cache miss

    def score_all(self, query: str) -> Dict[str, float]:
        """返回所有 turn 的向量相似度得分 {dia_id: cosine_sim}"""
        q_vec = self.embedder.encode(query, normalize=True)  # (384,)
        scores = self.matrix @ q_vec                          # (N,) cosine sim in [-1,1]
        return {dia_id: float(s) for dia_id, s in zip(self.dia_ids, scores)}

    def score_all_hyde(self, query: str, question: str = "") -> Dict[str, float]:
        """
        HyDE 向量得分：用原始问题生成假设文档，取均值向量后检索。

        query    — 预处理后的检索词（用于 fallback）
        question — 原始完整问题（用于 HyDE 模板匹配，必须是 wh-question 才有效）
        """
        # 用原始问题生成假设文档（模式匹配需要完整 wh-question）
        src = question if question else query
        hyp_texts = hyde_vector_queries(src)

        if len(hyp_texts) <= 1:
            return self.score_all(query)

        # 批量编码所有假设文档，取均值向量
        all_texts = hyp_texts  # 已包含预处理后的 query 作为第一项
        vecs = self.embedder.encode(all_texts, normalize=False)  # (n, 384) 批量
        avg_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec /= norm
        scores = self.matrix @ avg_vec      # (N,)
        return {dia_id: float(s) for dia_id, s in zip(self.dia_ids, scores)}

    def search(self, query: str, top_k: int,
               use_hyde: bool = False, question: str = "") -> List[str]:
        """返回 top-K 最相似的 dia_id 列表"""
        sc = (self.score_all_hyde(query, question)
              if use_hyde else self.score_all(query))
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        return [dia_id for dia_id, _ in ranked[:top_k]]


# ══════════════════════════════════════════════════════════════
# Hybrid 检索（FTS + 向量融合）
# ══════════════════════════════════════════════════════════════

def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max 归一化到 [0, 1]"""
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    return {k: (v - lo) / span for k, v in scores.items()}


def hybrid_search(
    query: str,
    turns: Dict[str, str],
    vector_index: "VectorIndex",
    top_k: int,
    alpha: float = 0.6,   # 向量权重，(1-alpha) 为 FTS 权重
    method: str = "linear",  # "linear" | "rrf"
    rrf_k: int = 60,
    bm25_key: str = "",
    use_hyde: bool = False,  # 是否对向量侧用 HyDE 均值 embedding
    fts_query: str = "",     # FTS 侧单独 query（不同于向量侧）
    question: str = "",      # 原始完整问题（用于 HyDE 模板匹配）
) -> List[str]:
    """
    Hybrid 检索：FTS + 向量得分融合。

    linear 模式（默认）：
        score = alpha * vec_norm + (1-alpha) * fts_norm
        两路分别 min-max 归一化后加权求和。

    rrf 模式（Reciprocal Rank Fusion）：
        score = 1/(rrf_k + rank_vec) + 1/(rrf_k + rank_fts)
        不依赖分数绝对值，对异常值鲁棒。
    """
    # ── 获取两路得分 ──────────────────────────────────────────
    vec_raw = (vector_index.score_all_hyde(query, question)
               if use_hyde else vector_index.score_all(query))   # cosine sim
    fts_q = fts_query if fts_query else query
    fts_raw = fts_scores(fts_q, turns, bm25_key)     # BM25

    all_ids = list(turns.keys())

    if method == "rrf":
        # RRF：按排名计算
        vec_ranked = sorted(all_ids, key=lambda x: -vec_raw.get(x, -1))
        fts_ranked = sorted(all_ids, key=lambda x: -fts_raw.get(x, 0))
        vec_rank = {dia_id: rank for rank, dia_id in enumerate(vec_ranked, 1)}
        fts_rank = {dia_id: rank for rank, dia_id in enumerate(fts_ranked, 1)}
        combined = {
            dia_id: 1.0 / (rrf_k + vec_rank[dia_id]) + 1.0 / (rrf_k + fts_rank[dia_id])
            for dia_id in all_ids
        }
    else:
        # Linear：min-max 归一化后加权
        # 向量得分：cosine sim 已经在 [-1,1]，直接归一化
        vec_norm = _minmax_normalize(vec_raw)
        # FTS 得分：词频可能全零（query 词不在任何 turn 中）
        fts_vals = list(fts_raw.values())
        if max(fts_vals) > 0:
            fts_norm = _minmax_normalize(fts_raw)
        else:
            fts_norm = {k: 0.0 for k in fts_raw}

        combined = {
            dia_id: alpha * vec_norm.get(dia_id, 0.0) + (1 - alpha) * fts_norm.get(dia_id, 0.0)
            for dia_id in all_ids
        }

    ranked = sorted(combined.items(), key=lambda x: -x[1])
    return [dia_id for dia_id, _ in ranked[:top_k]]


# ══════════════════════════════════════════════════════════════
# 主评测流程
# ══════════════════════════════════════════════════════════════

def run_evaluation(
    data: List[dict],
    method: str = "both",
    top_ks: List[int] = (1, 3, 5),
    category_filter: Optional[int] = None,
    sample_filter: Optional[int] = None,
    verbose: bool = False,
    cache_dir: Optional[Path] = None,
    hybrid_alpha: float = 0.6,
    hybrid_fusion: str = "linear",  # "linear" | "rrf"
    context_window: int = 0,
    use_query_preprocess: bool = True,
    use_iterative: bool = False,   # Cat3 多跳迭代检索
) -> dict:
    """
    method: "fts" | "vector" | "hybrid" | "all"
    hybrid_alpha: 向量权重（linear 模式）
    hybrid_fusion: "linear" 或 "rrf"
    use_query_preprocess: 是否对 query 做停用词过滤
    use_iterative: 是否对 Cat3 开启两轮迭代检索（第二轮用第一轮命中内容扩展 query）
    """
    use_fts    = method in ("fts", "all")
    use_vector = method in ("vector", "all")
    use_hybrid = method in ("hybrid", "all")
    # hybrid 需要向量模型
    need_embedder = use_vector or use_hybrid

    embedder = None
    if need_embedder:
        from embedder import Embedder
        print("[eval] 加载 Embedding 模型...", flush=True)
        t0 = time.perf_counter()
        embedder = Embedder()
        print(f"[eval] 模型加载完成：{(time.perf_counter()-t0)*1000:.0f}ms")

    # 汇总指标容器
    active_methods = (
        (["fts"] if use_fts else []) +
        (["vector"] if use_vector else []) +
        (["hybrid"] if use_hybrid else [])
    )
    results: Dict[str, Dict] = {}
    for m in active_methods:
        results[m] = {
            "recall": defaultdict(list),   # k → [per-qa recall]
            "mrr": [],
            "by_category": defaultdict(lambda: defaultdict(list)),
        }

    total_qa = 0
    skipped_qa = 0

    for sample_idx, sample in enumerate(data):
        if sample_filter is not None and sample_idx != sample_filter:
            continue

        sample_id = sample.get("sample_id", sample_idx)
        # FTS 用带 speaker+date 前缀的富文本（精确词匹配受益）
        turns_rich = extract_turns(sample, speaker_prefix=True, date_prefix=True)
        # 向量用干净原始文本（MiniLM 短窗口，前缀稀释有害）
        turns_clean = extract_turns(sample, speaker_prefix=False, date_prefix=False)
        # hybrid_search 的 fts_scores 用 rich，向量 score_all 用 clean
        # 统一接口：turns 指向 rich（FTS 用），向量索引内部用 clean
        turns = turns_rich  # FTS 入参

        if not turns:
            continue

        print(f"\n[样本 {sample_idx}] sample_id={sample_id}  turns={len(turns)}  QA={len(sample['qa'])}")

        # 建立向量索引（带缓存，hybrid 也需要）
        vector_index = None
        # BM25 索引（FTS 用富文本）
        bm25_key = f"bm25_{sample_id}"
        _get_bm25(turns_rich, bm25_key)  # 预热，后续查询直接复用

        if need_embedder:
            t0 = time.perf_counter()
            # 向量索引用干净文本（nosp），FTS 用富文本，两者 cache key 不同
            cache_key = f"sample_{sample_id}_nosp" if cache_dir else ""
            vector_index = VectorIndex(embedder, cache_dir=cache_dir)
            hit = vector_index.build(turns_clean, cache_key=cache_key, context_window=context_window)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            cache_flag = "✅ 缓存命中" if hit else "🔨 重新编码"
            print(f"  向量索引：{elapsed_ms:.0f}ms  ({len(turns)} turns × 384d)  [{cache_flag}]")

        # 构建 Observation 向量索引（Cat3 多跳推理专用）
        obs_matrix: Optional[np.ndarray] = None
        obs_dids: List[str] = []
        obs_texts: List[str] = []
        if need_embedder:
            obs_raw = sample.get("observation", {})
            obs_facts: List[tuple] = []
            for sess_obs in obs_raw.values():
                for facts in sess_obs.values():
                    if not isinstance(facts, list):
                        continue
                    for item in facts:
                        if isinstance(item, list) and len(item) == 2:
                            fact_text, did = item[0], item[1]
                            if isinstance(did, list):
                                did = did[0] if did else None
                            if did and isinstance(did, str) and isinstance(fact_text, str):
                                obs_facts.append((fact_text, did))
            if obs_facts:
                obs_texts = [f for f, _ in obs_facts]
                obs_dids  = [d for _, d in obs_facts]
                obs_matrix = embedder.encode(obs_texts, normalize=True)  # (N, 384)

        # 遍历 QA
        for qa in sample["qa"]:
            cat = qa.get("category", -1)
            if category_filter is not None and cat != category_filter:
                continue

            question = qa["question"]
            evidence = qa.get("evidence", [])

            # 跳过 category 5（对抗/无答案）
            if cat == 5:
                skipped_qa += 1
                continue

            if not evidence:
                skipped_qa += 1
                continue

            total_qa += 1
            max_k = max(top_ks)

            # Query 预处理：去掉疑问停用词，保留名词短语
            q_search = preprocess_query(question) if use_query_preprocess else question
            # HyDE 扩展：FTS 侧用属性词同义词扩展（Cat1 受益），向量侧用假设句均值
            q_fts = hyde_fts_expand(question) if use_query_preprocess else question

            if use_fts:
                retrieved_fts = fts_search(q_fts, turns, top_k=max_k,
                                           cache_key=bm25_key)
                for k in top_ks:
                    r = recall_at_k(retrieved_fts, evidence, k)
                    results["fts"]["recall"][k].append(r)
                    results["fts"]["by_category"][cat][k].append(r)
                results["fts"]["mrr"].append(reciprocal_rank(retrieved_fts, evidence))

            if use_vector:
                # Cat3 专项：observation 向量检索 + 扩展 query
                if cat == 3 and obs_matrix is not None:
                    q_vec = embedder.encode(q_search, normalize=True)
                    obs_sims = obs_matrix @ q_vec
                    top_idx = np.argsort(-obs_sims)[:4]
                    expanded_parts = [q_search]
                    cand_dids = []
                    for idx in top_idx:
                        expanded_parts.append(" ".join(obs_texts[idx].split()[:15]))
                        did = obs_dids[idx]
                        if did in turns_clean:
                            cand_dids.append(did)
                    expanded_q = " ".join(expanded_parts)
                    r2 = vector_index.search(expanded_q, top_k=max_k)
                    seen: set = set()
                    retrieved_vec = []
                    for did in cand_dids + r2:
                        if did not in seen and did in turns_rich:
                            seen.add(did)
                            retrieved_vec.append(did)
                    retrieved_vec = retrieved_vec[:max_k]
                elif use_iterative and cat == 3:
                    r1 = vector_index.search(q_search, top_k=3)
                    if r1 and r1[0] in turns_clean:
                        hit_text = turns_clean[r1[0]][:120]
                        expanded_q = q_search + " " + hit_text
                        r2 = vector_index.search(expanded_q, top_k=max_k)
                        seen = set()
                        retrieved_vec = []
                        for did in r2 + r1:
                            if did not in seen:
                                seen.add(did)
                                retrieved_vec.append(did)
                        retrieved_vec = retrieved_vec[:max_k]
                    else:
                        retrieved_vec = vector_index.search(q_search, top_k=max_k)
                else:
                    retrieved_vec = vector_index.search(q_search, top_k=max_k,
                                                        use_hyde=True, question=question)
                for k in top_ks:
                    r = recall_at_k(retrieved_vec, evidence, k)
                    results["vector"]["recall"][k].append(r)
                    results["vector"]["by_category"][cat][k].append(r)
                results["vector"]["mrr"].append(reciprocal_rank(retrieved_vec, evidence))

            if use_hybrid:
                # Cat3 专项：observation 向量检索 + 扩展 query 后做 hybrid
                if cat == 3 and obs_matrix is not None:
                    q_vec = embedder.encode(q_search, normalize=True)
                    obs_sims = obs_matrix @ q_vec
                    top_idx = np.argsort(-obs_sims)[:4]
                    expanded_parts = [q_search]
                    cand_dids = []
                    for idx in top_idx:
                        expanded_parts.append(" ".join(obs_texts[idx].split()[:15]))
                        did = obs_dids[idx]
                        if did in turns_rich:
                            cand_dids.append(did)
                    expanded_q = " ".join(expanded_parts)
                    r2 = hybrid_search(expanded_q, turns, vector_index,
                                       top_k=max_k, bm25_key=bm25_key)
                    seen_h: set = set()
                    retrieved_hyb = []
                    for did in cand_dids + r2:
                        if did not in seen_h and did in turns_rich:
                            seen_h.add(did)
                            retrieved_hyb.append(did)
                    retrieved_hyb = retrieved_hyb[:max_k]
                elif use_iterative and cat == 3:
                    r1 = hybrid_search(q_search, turns, vector_index, top_k=3,
                                       alpha=hybrid_alpha, method=hybrid_fusion,
                                       bm25_key=bm25_key)
                    if r1 and r1[0] in turns_clean:
                        hit_text = turns_clean[r1[0]][:120]
                        expanded_q = q_search + " " + hit_text
                        r2 = hybrid_search(expanded_q, turns, vector_index,
                                           top_k=max_k, alpha=hybrid_alpha,
                                           method=hybrid_fusion, bm25_key=bm25_key)
                        seen = set()
                        retrieved_hyb = []
                        for did in r2 + r1:
                            if did not in seen:
                                seen.add(did)
                                retrieved_hyb.append(did)
                        retrieved_hyb = retrieved_hyb[:max_k]
                    else:
                        retrieved_hyb = hybrid_search(
                            q_search, turns, vector_index,
                            top_k=max_k, alpha=hybrid_alpha,
                            method=hybrid_fusion, bm25_key=bm25_key,
                            use_hyde=True, fts_query=q_fts, question=question)
                else:
                    retrieved_hyb = hybrid_search(
                        q_search, turns, vector_index,
                        top_k=max_k,
                        alpha=hybrid_alpha,
                        method=hybrid_fusion,
                        bm25_key=bm25_key,
                        use_hyde=True,
                        fts_query=q_fts,
                        question=question,
                    )
                for k in top_ks:
                    r = recall_at_k(retrieved_hyb, evidence, k)
                    results["hybrid"]["recall"][k].append(r)
                    results["hybrid"]["by_category"][cat][k].append(r)
                results["hybrid"]["mrr"].append(reciprocal_rank(retrieved_hyb, evidence))

            if verbose:
                print(f"  Q: {question[:60]!r}")
                print(f"     GT: {evidence}")
                if use_fts:
                    print(f"     FTS    top3: {retrieved_fts[:3]}")
                if use_vector:
                    print(f"     VEC    top3: {retrieved_vec[:3]}")
                if use_hybrid:
                    print(f"     HYBRID top3: {retrieved_hyb[:3]}")

    # ── 汇总 ─────────────────────────────────────────────────
    summary = {
        "total_qa": total_qa,
        "skipped_qa": skipped_qa,
        "top_ks": list(top_ks),
        "methods": {},
    }

    for m, res in results.items():
        method_summary = {
            "recall": {f"R@{k}": float(np.mean(res["recall"][k])) if res["recall"][k] else 0.0
                       for k in top_ks},
            "mrr": float(np.mean(res["mrr"])) if res["mrr"] else 0.0,
            "by_category": {},
        }
        for cat, cat_res in res["by_category"].items():
            method_summary["by_category"][str(cat)] = {
                "name": CATEGORY_NAMES.get(cat, f"Cat{cat}"),
                "n": len(cat_res[top_ks[0]]),
                "recall": {f"R@{k}": float(np.mean(cat_res[k])) if cat_res[k] else 0.0
                           for k in top_ks},
            }
        summary["methods"][m] = method_summary

    return summary


# ══════════════════════════════════════════════════════════════
# 报告输出
# ══════════════════════════════════════════════════════════════

def print_report(summary: dict):
    top_ks = summary["top_ks"]
    print("\n" + "=" * 65)
    print(f"  LoCoMo-10 向量检索评测报告")
    print(f"  有效 QA: {summary['total_qa']}  跳过: {summary['skipped_qa']}")
    print("=" * 65)

    METHOD_LABELS = {
        "fts":    "📝 FTS 关键词基线",
        "vector": "🔍 向量语义检索",
        "hybrid": "⚡ Hybrid 融合检索",
    }

    for method, res in summary["methods"].items():
        label = METHOD_LABELS.get(method, method)
        print(f"\n{label}")
        print("  " + "-" * 50)

        recall_strs = "  ".join(
            f"R@{k}={res['recall'][f'R@{k}']:.3f}" for k in top_ks
        )
        print(f"  总体:  {recall_strs}   MRR={res['mrr']:.3f}")

        print("\n  按 Category：")
        for cat_str in sorted(res["by_category"].keys(), key=int):
            cat_res = res["by_category"][cat_str]
            n = cat_res["n"]
            name = cat_res["name"]
            r_strs = "  ".join(
                f"R@{k}={cat_res['recall'][f'R@{k}']:.3f}" for k in top_ks
            )
            print(f"    Cat{cat_str} ({n:4d}个) {name[:22]:25s}: {r_strs}")

    # 对比提升
    def _print_delta(label: str, base_key: str, target_key: str):
        if base_key not in summary["methods"] or target_key not in summary["methods"]:
            return
        base_r = summary["methods"][base_key]["recall"]
        tgt_r  = summary["methods"][target_key]["recall"]
        print(f"\n  📊 {label}：")
        for k in top_ks:
            key = f"R@{k}"
            delta = tgt_r[key] - base_r[key]
            sign = "+" if delta >= 0 else ""
            print(f"    R@{k}: {base_r[key]:.3f} → {tgt_r[key]:.3f}  ({sign}{delta:.3f})")
        b_mrr = summary["methods"][base_key]["mrr"]
        t_mrr = summary["methods"][target_key]["mrr"]
        dm = t_mrr - b_mrr
        print(f"    MRR: {b_mrr:.3f} → {t_mrr:.3f}  ({'+'if dm>=0 else ''}{dm:.3f})")

    _print_delta("向量 vs FTS",   "fts",    "vector")
    _print_delta("Hybrid vs FTS", "fts",    "hybrid")
    _print_delta("Hybrid vs 向量","vector", "hybrid")

    print("=" * 65)


# ══════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LoCoMo-10 检索评测")
    parser.add_argument("--method", choices=["fts", "vector", "hybrid", "all"], default="all")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5],
                        help="评测的 K 值列表（默认: 1 3 5）")
    parser.add_argument("--category", type=int, default=None,
                        help="只评测指定 category（1-5）")
    parser.add_argument("--sample-id", type=int, default=None,
                        help="只评测第 N 个样本（0-indexed）")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每条 QA 的检索结果")
    parser.add_argument("--output", type=str, default=None,
                        help="将 JSON 报告写入文件")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="向量索引缓存目录（指定后首次编码写入磁盘，后续直接加载）")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Hybrid 向量权重，范围 0~1（默认 0.6）")
    parser.add_argument("--fusion", choices=["linear", "rrf"], default="linear",
                        help="Hybrid 融合策略：linear 加权求和 / rrf 排名融合（默认 linear）")
    parser.add_argument("--window", type=int, default=1,
                        help="上下文扩展窗口：0=不扩展，1=前后各1条，2=前后各2条（默认 1）")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="禁用 query 预处理（停用词过滤）")
    parser.add_argument("--iterative", action="store_true",
                        help="对 Cat3 多跳问题开启两轮迭代检索")
    args = parser.parse_args()

    data = load_dataset(DATA_PATH)
    print(f"[eval] 数据集加载完成：{len(data)} 个样本，共 {sum(len(d['qa']) for d in data)} 个 QA")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] 向量缓存目录：{cache_dir}")

    t_start = time.perf_counter()
    summary = run_evaluation(
        data=data,
        method=args.method,
        top_ks=sorted(set(args.top_k)),
        category_filter=args.category,
        sample_filter=args.sample_id,
        verbose=args.verbose,
        cache_dir=cache_dir,
        hybrid_alpha=args.alpha,
        hybrid_fusion=args.fusion,
        context_window=args.window,
        use_query_preprocess=not args.no_preprocess,
        use_iterative=args.iterative,
    )
    elapsed = time.perf_counter() - t_start

    print_report(summary)
    print(f"\n  总耗时：{elapsed:.1f}s")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  报告已写入：{out_path}")


if __name__ == "__main__":
    main()
