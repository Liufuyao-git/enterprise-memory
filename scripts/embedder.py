#!/usr/bin/env python3
"""
embedder.py — ONNX Embedding 工具类
基于 all-MiniLM-L6-v2（BERT 架构，384维输出）

功能：
  - tokenize + ONNX 推理 + mean pooling + L2 归一化
  - 支持单条 / 批量 encode
  - 纯 CPU 推理，零 torch 依赖

用法（CLI）：
  python3 embedder.py "你好世界"
  python3 embedder.py "文本1" "文本2" "文本3"
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Union

# ── 默认模型路径（相对于本文件的父目录）─────────────────────
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2"
_DEFAULT_ONNX_PATH = _DEFAULT_MODEL_DIR / "onnx" / "model.onnx"
_DEFAULT_TOKENIZER_PATH = _DEFAULT_MODEL_DIR / "tokenizer.json"

# BERT 默认参数
_MAX_LENGTH = 512
_PAD_TOKEN_ID = 0


class Embedder:
    """
    ONNX-based sentence embedder.

    参数:
        onnx_path:      ONNX 模型文件路径
        tokenizer_path: tokenizer.json 路径（HuggingFace fast tokenizer 格式）
        max_length:     最大序列长度（截断/填充到此长度）
        batch_size:     批量推理时的分块大小（避免 OOM）
    """

    def __init__(
        self,
        onnx_path: Union[str, Path] = _DEFAULT_ONNX_PATH,
        tokenizer_path: Union[str, Path] = _DEFAULT_TOKENIZER_PATH,
        max_length: int = _MAX_LENGTH,
        batch_size: int = 32,
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.max_length = max_length
        self.batch_size = batch_size

        # 加载 tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(
            pad_id=_PAD_TOKEN_ID,
            pad_token="[PAD]",
            length=max_length,
        )
        self._tokenizer.enable_truncation(max_length=max_length)

        # 加载 ONNX session（CPU，无日志噪音）
        so = ort.SessionOptions()
        so.log_severity_level = 3  # ERROR only
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        # 验证输入输出名（兼容不同导出方式）
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name

    # ── 公开 API ──────────────────────────────────────────────

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        编码一条或多条文本，返回 (N, 384) float32 ndarray。
        normalize=True 时对输出做 L2 归一化（余弦相似度场景必须开启）。
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            emb = self._encode_batch(batch)
            embeddings.append(emb)

        result = np.concatenate(embeddings, axis=0)  # (N, 384)
        if normalize:
            result = self._l2_normalize(result)

        return result[0] if single else result

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个已归一化向量的余弦相似度。
        a, b 可以是 (384,) 或 (1, 384)。
        """
        a = a.flatten()
        b = b.flatten()
        return float(np.dot(a, b))

    # ── 内部方法 ──────────────────────────────────────────────

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """对一个 batch 做 tokenize + ONNX 推理 + mean pooling"""
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # 构造 ONNX 输入（兼容不同导出时的输入名）
        feed = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = token_type_ids

        # 推理：取第一个输出（last_hidden_state 或 token_embeddings）
        # shape: (batch, seq_len, hidden_size)
        output = self._session.run([self._output_name], feed)[0]

        # Mean pooling（仅对非 PAD token 取均值）
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)  # (B, L, 1)
        summed = (output * mask).sum(axis=1)                         # (B, H)
        counts = mask.sum(axis=1).clip(min=1e-9)                     # (B, 1)
        return (summed / counts).astype(np.float32)                  # (B, H)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        """逐行 L2 归一化"""
        norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)
        return x / norms


# ── 模块级单例（lazy init，避免多次加载模型）─────────────────

_instance: Embedder | None = None


def get_embedder(
    onnx_path: Union[str, Path] = _DEFAULT_ONNX_PATH,
    tokenizer_path: Union[str, Path] = _DEFAULT_TOKENIZER_PATH,
) -> Embedder:
    """
    返回全局 Embedder 单例。
    首次调用时初始化，后续调用直接复用。
    """
    global _instance
    if _instance is None:
        _instance = Embedder(onnx_path=onnx_path, tokenizer_path=tokenizer_path)
    return _instance


def embed(
    texts: Union[str, List[str]],
    normalize: bool = True,
) -> np.ndarray:
    """
    快捷函数：使用默认路径的全局 Embedder 编码文本。
    适合在其他模块中直接 `from embedder import embed` 使用。
    """
    return get_embedder().encode(texts, normalize=normalize)


# ── CLI ───────────────────────────────────────────────────────

def _cli():
    import time

    texts = sys.argv[1:]
    if not texts:
        print("用法：python3 embedder.py <文本1> [文本2] ...")
        sys.exit(1)

    print(f"[embedder] 加载模型...", file=sys.stderr)
    t0 = time.perf_counter()
    embedder = Embedder()
    t1 = time.perf_counter()
    print(f"[embedder] 模型加载完成：{(t1-t0)*1000:.1f}ms", file=sys.stderr)

    print(f"[embedder] 编码 {len(texts)} 条文本...", file=sys.stderr)
    t2 = time.perf_counter()
    embeddings = embedder.encode(texts)
    t3 = time.perf_counter()
    print(f"[embedder] 推理完成：{(t3-t2)*1000:.1f}ms", file=sys.stderr)

    # 输出结果
    if len(texts) == 1:
        vec = embeddings
        print(f"\n文本：{texts[0]!r}")
        print(f"维度：{vec.shape}")
        print(f"范数：{np.linalg.norm(vec):.6f}")
        print(f"前8维：{vec[:8].tolist()}")
    else:
        print(f"\n批量编码结果（shape: {embeddings.shape}）：")
        for i, (text, vec) in enumerate(zip(texts, embeddings)):
            short = repr(text[:40])
            print(f"  [{i}] {short:45s} norm={np.linalg.norm(vec):.4f}")

        # 顺便打印相似度矩阵
        print("\n余弦相似度矩阵：")
        n = len(texts)
        for i in range(n):
            row = []
            for j in range(n):
                sim = embedder.similarity(embeddings[i], embeddings[j])
                row.append(f"{sim:.3f}")
            print(f"  {texts[i][:20]!r:25s}: {' '.join(row)}")


if __name__ == "__main__":
    _cli()
