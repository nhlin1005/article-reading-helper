# -*- coding: utf-8 -*-
"""
封装：加载（或自动训练）关键词抽取模型，
并提供对一段文本抽取关键词的函数。

这一版修正了 BERT wordpiece 的还原逻辑：
- 会把 "chi", "##nese", "buddhist", "monk"
  还原成 "chinese buddhist monk"
- 不会再出现 "chinesebuddhistmonk" 这种连在一起的怪词
"""

import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from config import MODEL_DIR, MAX_SEQ_LEN, DEFAULT_TOP_N, MIN_TOKEN_LEN

_model = None
_tokenizer = None
_device = None


def _ensure_model_loaded():
    """懒加载模型：第一次调用时才真正 from_pretrained。"""
    global _model, _tokenizer, _device
    if _model is not None:
        return

    if not (os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR)):
        # 如果没有模型目录，则触发训练（一般只在服务器跑一次）
        from train_keyword_model import train
        print("⚠️ 模型目录不存在，将先训练一个模型...")
        train()

    print(f"🔌 Loading keyword model from: {MODEL_DIR}")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(_device)
    _model.eval()


def _merge_wordpieces_to_phrase(pieces: List[str]) -> str:
    """
    把 BERT 的 wordpieces 合成可读的短语：
    - ["chi", "##nese"] -> "chinese"
    - ["chinese", "buddhist", "monk"] -> "chinese buddhist monk"
    """
    words: List[str] = []
    for p in pieces:
        if p.startswith("##"):
            sub = p[2:]
            if not words:
                words.append(sub)
            else:
                words[-1] = words[-1] + sub
        else:
            words.append(p)
    return " ".join(words)


def _extract_spans_from_tokens(
    tokens: List[str],
    label_ids: List[int],
    id2label: dict
) -> List[str]:
    """
    根据 token 标签序列（B/I/O）拼出关键词 span。
    这里会保留词之间的空格，不会再出现 chinesebuddhistmonk 这种情况。
    """
    keywords: List[str] = []
    current_pieces: List[str] = []

    special_tokens = set(
        [
            getattr(_tokenizer, "cls_token", "[CLS]"),
            getattr(_tokenizer, "sep_token", "[SEP]"),
            getattr(_tokenizer, "pad_token", "[PAD]"),
        ]
    )

    for tok, lid in zip(tokens, label_ids):
        label = id2label.get(int(lid), "O")

        # 跳过特殊 token
        if tok in special_tokens or tok in _tokenizer.all_special_tokens:
            if current_pieces:
                phrase = _merge_wordpieces_to_phrase(current_pieces)
                keywords.append(phrase)
                current_pieces = []
            continue

        if label == "B":
            # 开启新的短语
            if current_pieces:
                phrase = _merge_wordpieces_to_phrase(current_pieces)
                keywords.append(phrase)
            current_pieces = [tok]
        elif label == "I" and current_pieces:
            current_pieces.append(tok)
        else:
            # O 或不合理的 I：结束当前 span
            if current_pieces:
                phrase = _merge_wordpieces_to_phrase(current_pieces)
                keywords.append(phrase)
                current_pieces = []

    # 收尾
    if current_pieces:
        phrase = _merge_wordpieces_to_phrase(current_pieces)
        keywords.append(phrase)

    # 简单清理 + 去重 + 过滤太短的垃圾 span（比如 "b"）
    cleaned: List[str] = []
    for k in keywords:
        k = k.replace("  ", " ").strip().lower()
        if not k:
            continue
        if len(k) < 2:   # 丢掉特别短的
            continue
        if k not in cleaned:
            cleaned.append(k)

    return cleaned


def extract_keywords_from_text(text: str,
                               top_n: int = DEFAULT_TOP_N) -> List[str]:
    """
    直接对一段英文文本跑模型，
    返回模型认为是关键短语的若干候选（短语形式，比如 "chinese buddhist monk"）。
    注意：在 AI 选词流程里会再把短语拆成单词。
    """
    _ensure_model_loaded()
    tokenizer = _tokenizer
    model = _model
    device = _device

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits  # [1, seq_len, num_labels]
    pred_ids = logits.argmax(-1).squeeze(0).tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    id2label = model.config.id2label

    spans = _extract_spans_from_tokens(tokens, pred_ids, id2label)

    if top_n and len(spans) > top_n:
        spans = spans[:top_n]

    return spans
