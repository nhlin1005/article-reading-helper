# -*- coding: utf-8 -*-
"""
可选：用大语言模型对候选词做进一步筛选、排序。
目前如果没有设置 OPENAI_API_KEY，就直接原样返回，不影响流程。

新增：
- generate_meaning_example_with_context(word, context)：
  当字典查不到释义时，喂文章上下文给 LLM 生成 “meaning + example”（JSON）。
"""

import os
import json
from typing import List, Optional, Dict, Any

from config import USE_LLM_REFINER

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")


def refine_keywords_with_llm(
    article_text: str,
    candidates: List[str],
    top_n: int
) -> List[str]:
    """
    如果有 LLM，就调用 LLM；否则就简单截断。
    这里为了方便部署，默认实现是「不真连 API」。
    你将来可以在这里接自己的 RankLLaVA / Qwen / OpenAI 等。
    """
    if not USE_LLM_REFINER or not _OPENAI_KEY:
        return candidates[:top_n]

    # 你目前没有真正的排序需求，就先保持原逻辑（不破坏）
    return candidates[:top_n]


def _openai_chat_json(prompt: str, model: str = "gpt-4o-mini", timeout: int = 60) -> Optional[Dict[str, Any]]:
    """
    尝试调用 OpenAI Chat Completions，并要求返回 JSON。
    如果依赖/网络/Key 不可用，返回 None（不影响主流程）。
    """
    if not _OPENAI_KEY:
        return None

    # 兼容不同安装方式：优先用 openai SDK；没有就失败返回 None
    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = OpenAI(api_key=_OPENAI_KEY)

        # 要求模型严格输出 JSON
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful English vocabulary tutor. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=timeout,
        )

        text = resp.choices[0].message.content
        if not text:
            return None

        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def generate_meaning_example_with_context(
    word: str,
    context: str,
    model: str = "gpt-4o-mini"
) -> Optional[Dict[str, str]]:
    """
    当字典查不到 meaning/example 时：
    喂文章上下文给 LLM 生成：
      - meaning: 1 句英文释义（不要 markdown）
      - example: 1 句英文例句（符合上下文，避免编造事实）

    返回 {"meaning": "...", "example": "..."} 或 None（表示不启用/失败）。
    """
    if not USE_LLM_REFINER or not _OPENAI_KEY:
        return None
    if not context:
        return None

    prompt = f"""
Task:
Given a target word and its surrounding article context, write:
1) meaning: one short sentence defining the word as used in THIS context
2) example: one natural English example sentence consistent with the context

Rules:
- If it's a proper noun/name, say so and explain what it refers to in this context.
- Do NOT invent facts not supported by the context.
- Output ONLY JSON with keys: meaning, example.

Target word: "{word}"
Context: "{context}"
""".strip()

    obj = _openai_chat_json(prompt, model=model, timeout=60)
    if not obj:
        return None

    meaning = str(obj.get("meaning", "")).strip()
    example = str(obj.get("example", "")).strip()

    if not meaning and not example:
        return None

    return {"meaning": meaning, "example": example}
