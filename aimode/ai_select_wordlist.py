# -*- coding: utf-8 -*-
"""
ai_select_wordlist.py

根据文章内容 + 已训练好的关键词模型，自动挑选一批“可能是生词”的英文单词。

特性：
- 使用 DistilBERT 关键词模型 + 规则 + 词频
- 尽量包括 succession / virtues / robes / moustache / generations 这类词
- 尽量排除 four / lake / river / little 这类太简单的高频词
- 尽量排除 shwo-yih-tsai-yu-po / sarvistivadas 这类音译怪词
- 支持 top_n 为“数量”（>=1）或“比例”（0~1）
"""

from typing import List, Dict, Set

from config import DEFAULT_TOP_N, MAX_CANDIDATES_BEFORE_LLM, MIN_TOKEN_LEN
from text_utils import get_word_freqs, tokenize_text_to_words
from candidate_builder import build_candidates, COMMON_WORDS as BASE_COMMON_WORDS
from keyword_extractor import extract_keywords_from_text
from llm_refiner import refine_keywords_with_llm


# 在原有 COMMON_WORDS 基础上额外补一批太简单但经常被选中的词
EXTRA_COMMON: Set[str] = {
    "four",
    "great",
    "little",
    "like",
    "people",
    "country",
    "land",
    "lake",
    "river",
    "north",
    "south",
    "east",
    "west",
    "called",
}

COMMON_WORDS: Set[str] = set(BASE_COMMON_WORDS) | EXTRA_COMMON


def _normalize_token(tok: str) -> str:
    """小写 + 去掉两端标点符号。"""
    tok = tok.strip().lower()
    return tok.strip(".,;:!?()[]{}\"'`")


def _is_weird_token(tok: str) -> bool:
    """
    判断是不是“很奇怪、不像正常英文单词”的 token：
    - 含数字
    - 含非 ASCII 字符
    - 连字符/单引号太多（音译/特殊记号）
    - 又有连字符又有引号
    - 太长且几乎没有元音
    这类我们要么直接丢掉，要么排到最后。
    """
    if not tok:
        return True

    # 有数字
    if any(ch.isdigit() for ch in tok):
        return True

    # 非 ASCII 字符
    try:
        tok.encode("ascii")
    except UnicodeEncodeError:
        return True

    # 连字符/引号过多
    hy = tok.count("-")
    ap = tok.count("'")
    if hy > 1 or ap > 1:
        return True
    if hy >= 1 and ap >= 1:
        return True

    # 特别长又几乎没元音（大概率是缩写/音译）
    if len(tok) > 15 and not any(v in tok for v in "aeiou"):
        return True

    return False


def _decide_n_to_select(total: int, top_n: float | int | None) -> int:
    """
    根据 total（候选词总数）和 top_n 决定最终选多少个。

    约定：
      - top_n is None: 使用 DEFAULT_TOP_N（如果 <1 则视为比例）
      - top_n >= 1: 视为“具体数量”
      - 0 < top_n < 1: 视为“比例”，例如 0.1 = 10%
    """
    if total <= 0:
        return 0

    if top_n is None:
        value = DEFAULT_TOP_N
    else:
        value = top_n

    # DEFAULT_TOP_N/传入值 如果 < 1，当成比例
    if isinstance(value, float) and value < 1.0:
        n = int(total * value)
    elif value < 1:  # int 且为 0 的兜底
        n = int(total * 0.1)
    else:
        n = int(value)

    # 合理边界
    if n < 1:
        n = 1
    if n > total:
        n = total
    return n


def ai_select_words_for_article(text: str,
                                top_n: float | int = DEFAULT_TOP_N) -> List[str]:
    """
    主函数：给一整篇英文文章，返回 top_n 个“可能是生词”的单词（小写形式）。

    top_n:
      - >=1: 视为“要几个词”（例如 30）
      - 0~1: 视为“比例”（例如 0.1 = 10%）
    """

    # 1) 词频统计
    freq: Dict[str, int] = get_word_freqs(text)

    # 2) 规则候选
    rule_candidates: List[str] = build_candidates(
        text,
        max_candidates=MAX_CANDIDATES_BEFORE_LLM,
    )

    # 3) 模型关键词 span
    raw_spans: List[str] = extract_keywords_from_text(
        text,
        top_n=MAX_CANDIDATES_BEFORE_LLM,
    )

    # 把 span 拆成词：模型特别关注的词集合
    model_keyword_words: Set[str] = set()
    for span in raw_spans:
        for w in tokenize_text_to_words(span):
            tok = _normalize_token(w)
            if not tok:
                continue
            if len(tok) < MIN_TOKEN_LEN:
                continue
            if tok in COMMON_WORDS:
                continue
            if _is_weird_token(tok):
                continue
            model_keyword_words.add(tok)

    # 4) 构建候选池（增强召回）
    candidates: Set[str] = set()

    # 4.0 全文中所有“正常、够长、非常见”的词
    for w, c in freq.items():
        tok = _normalize_token(w)
        if not tok:
            continue
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in COMMON_WORDS:
            continue
        if _is_weird_token(tok):
            continue
        candidates.add(tok)

    # 4.1 规则候选
    for w in rule_candidates:
        tok = _normalize_token(w)
        if not tok:
            continue
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in COMMON_WORDS:
            continue
        if _is_weird_token(tok):
            continue
        candidates.add(tok)

    # 4.2 模型关键词拆词
    for kw in model_keyword_words:
        candidates.add(kw)

    if not candidates:
        return []

    # 兜底：freq 里没有的，给个 1
    for w in list(candidates):
        if w not in freq:
            freq[w] = 1

    # 5) 排序打分：
    #   1. weird token 尽量放最后
    #   2. 词频越低越靠前
    #   3. 模型关键词稍微加分
    #   4. 词越长越靠前
    #   5. 字母序兜底
    def score(token: str):
        weird = 1 if _is_weird_token(token) else 0
        token_freq = freq.get(token, 10**9)
        is_model_word = 0 if token in model_keyword_words else 1
        return (
            weird,          # 0 正常词优先，1 怪词放后面
            token_freq,     # 越小越好（越少见越靠前）
            is_model_word,  # 0 模型关键词略优
            -len(token),    # 越长越好
            token,
        )

    ordered = sorted(candidates, key=score)

    # 6) 根据“数量 or 比例”决定最终要几个
    total = len(ordered)
    n_select = _decide_n_to_select(total, top_n)

    # 7) （可选）LLM 精修
    refined = refine_keywords_with_llm(text, ordered, top_n=n_select)

    # 8) 最终截断 + 再过滤一层
    out: List[str] = []
    seen = set()

    for w in refined:
        tok = _normalize_token(w)
        if not tok:
            continue
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in COMMON_WORDS:
            continue
        if _is_weird_token(tok):
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
        if len(out) >= n_select:
            break

    return out


if __name__ == "__main__":
    demo_text = (
        "The succession of emperors was recorded in the annals, "
        "and their robes symbolized virtues and authority across generations. "
        "He had a long moustache and very strict regulations to follow."
    )
    words_abs = ai_select_words_for_article(demo_text, top_n=10)
    words_ratio = ai_select_words_for_article(demo_text, top_n=0.5)
    print("Selected (top_n=10):", words_abs)
    print("Selected (top_n=0.5):", words_ratio)
