# -*- coding: utf-8 -*-
"""
rule-based 候选词生成：
- 去掉 very 常见的高频词（a, the, of, in, ...）
- 保留长度 ≥ MIN_TOKEN_LEN 的词
- 按出现顺序 + 一点 frequency 排序，保证 recall 较高
"""

from typing import List
from collections import Counter

from config import MIN_TOKEN_LEN, MAX_CANDIDATES_BEFORE_LLM
from text_utils import tokenize_text_to_words, get_word_freqs

# 一个简单的「常见英文词」表，用来排除太基础的词
COMMON_WORDS = {
    "a", "an", "the",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "them", "us",
    "my", "your", "his", "their", "our",
    "this", "that", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "can", "could", "may", "might", "must", "should", "will", "would",
    "of", "in", "on", "at", "by", "for", "from", "to", "with", "without",
    "and", "or", "but", "if", "so", "because",
    "as", "than", "then", "too", "very", "also",
    "not", "no", "yes",
    "there", "here", "when", "where", "why", "how",
    "who", "whom", "which", "what",
    "up", "down", "out", "over", "under",
    "into", "through", "between", "among", "about",
}


def build_candidates(text: str,
                     max_candidates: int = MAX_CANDIDATES_BEFORE_LLM) -> List[str]:
    """
    从整篇文章里找出「可能是生词」的候选列表（不考虑顺序和难度，只要 recall 高）：
      - 去掉常见小词
      - 长度 >= MIN_TOKEN_LEN
      - 保留重复出现次数较多的词
    """
    tokens = tokenize_text_to_words(text)
    freqs: Counter = get_word_freqs(text)

    seen = set()
    candidates: List[str] = []

    for tok in tokens:
        if tok in seen:
            continue
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in COMMON_WORDS:
            continue

        seen.add(tok)
        candidates.append(tok)

    # 简单按「频率高 + 长度长」排序，让重要词更靠前
    candidates.sort(key=lambda w: (-freqs.get(w, 1), -len(w), w))

    if max_candidates and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    return candidates
