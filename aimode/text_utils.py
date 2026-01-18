# -*- coding: utf-8 -*-
"""
文本工具函数：分词、frequency 统计。
尽量复用 build_vocab_combined 里的 normalize_token，保持一致。
"""

import re
import string
from collections import Counter
from typing import List

try:
    # 优先用你原来 build_vocab_combined 的 normalize_token
    from build_vocab_combined import normalize_token as _normalize_token_existing
except ImportError:
    _normalize_token_existing = None

PUNCTUATION = string.punctuation
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")


def _local_normalize_token(tok: str) -> str:
    """备选版 normalize：和你原来的逻辑接近。"""
    tok = tok.strip().strip(PUNCTUATION)
    m = WORD_RE.match(tok)
    return m.group(0).lower() if m else tok.lower()


def normalize_token(tok: str) -> str:
    if _normalize_token_existing is not None:
        return _normalize_token_existing(tok)
    return _local_normalize_token(tok)


def tokenize_text_to_words(text: str) -> List[str]:
    """把一段纯文本切成小写单词列表（只保留有意义的 token）。"""
    out: List[str] = []
    for raw in text.split():
        tok = normalize_token(raw)
        if tok:
            out.append(tok)
    return out


def get_word_freqs(text: str) -> Counter:
    """统计文本里单词频率。"""
    toks = tokenize_text_to_words(text)
    return Counter(toks)
