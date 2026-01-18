# -*- coding: utf-8 -*-
"""
可选：用大语言模型对候选词做进一步筛选、排序。
目前如果没有设置 OPENAI_API_KEY，就直接原样返回，不影响流程。
"""

import os
from typing import List

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
        # 简单策略：直接取前 top_n 个候选
        return candidates[:top_n]

    # TODO: 在这里接你的 LLM 推理逻辑
    # 伪代码示意：
    # prompt = f"..."
    # response = openai.chat.completions.create(...)
    # parsed = ...
    # return parsed

    return candidates[:top_n]
