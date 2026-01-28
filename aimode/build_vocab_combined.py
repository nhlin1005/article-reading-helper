# -*- coding: utf-8 -*-
"""
Builds a vocabulary CSV from an article + a selection list of words.

Defaults for your folder:
  --article     sat.txt
  --select      20241226.txt
  --out_words   satword.txt
  --out_csv     words.csv

Outputs:
  - satword.txt        (one word per line; order of first appearance)
  - words.csv          (word, meaning-from-Webster, example-from-Cambridge OR article context)

Rules:
  - MEANING from Merriam-Webster
  - EXAMPLE from Cambridge Dictionary
  - 如果字典里没有例句，则从原文中自动找一条包含该词的句子作为例句
  - 如果连词典释义也没有，则给一个兜底说明（可能是专有名词/音译），再尽量附上原文句子
  - 不再输出 source 列

另外：
  - 丢弃所有包含 `'` 的单词（例如 xuanzang's）
  - 丢弃所有含有 >2 个 '-' 的单词（例如 shwo-yih-tsai-yu-po）
  - 对英式拼写（含 "our"）尝试转成美式 ("or") 再去 Merriam-Webster 查一次

  python build_vocab_combined.py --article article.txt --select 20241226.txt --out_words satword.txt --out_csv words.csv
"""

import argparse
import csv
import time
import string
import re
import random
from typing import Optional, List, Iterable

import requests
from lxml import html
from urllib.parse import quote


import functools
print = functools.partial(print, flush=True)

# ----------------------------
# Live progress for front-end polling
# ----------------------------
# NOTE: server.py imports this module and exposes PROGRESS at /api/progress.
PROGRESS = {
    "status": "idle",   # idle | running | done
    "current": 0,
    "total": 0,
    "word": "",
}


def _set_progress(*, status: str = None, current: int = None, total: int = None, word: str = None) -> None:
    try:
        if status is not None:
            PROGRESS["status"] = status
        if current is not None:
            PROGRESS["current"] = int(current)
        if total is not None:
            PROGRESS["total"] = int(total)
        if word is not None:
            PROGRESS["word"] = word
    except Exception:
        pass

# ----------------------------
# 1) Tokenization / normalization
# ----------------------------
PUNCTUATION = string.punctuation
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")


def normalize_token(tok: str) -> str:
    """Lowercase, strip punctuation on the ends; keep inner hyphens/apostrophes."""
    tok = tok.strip().strip(PUNCTUATION)
    m = WORD_RE.match(tok)
    return m.group(0).lower() if m else tok.lower()


def tokenize_file(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            for raw in line.split():
                tok = normalize_token(raw)
                if tok:
                    out.append(tok)
    return out


# ----------------------------
# 2) HTTP helpers
# ----------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})
TIMEOUT = 15


def get_tree(url: str):
    try:
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return html.fromstring(r.text)
    except requests.RequestException:
        return None


# ----------------------------
# 3) Merriam-Webster meaning (带英式→美式 fallback)
# ----------------------------
def _webster_meaning_core(word: str) -> Optional[str]:
    """真正发请求 + 解析 Merriam-Webster 页面。"""
    url = f"https://www.merriam-webster.com/dictionary/{word}"
    tree = get_tree(url)
    if tree is None:
        return None

    def_nodes = tree.xpath(
        "//div[contains(@id,'dictionary-entry')]/div[@class='vg']"
        "//span[@class='dtText' or @class='unText']"
    )
    for n in def_nodes:
        txt = html.tostring(n, encoding="unicode")
        # 去 HTML 标签、前面的冒号
        txt = re.sub(r"<.*?>", "", txt)
        txt = re.sub(r"^\s*:\s*", "", txt).strip()
        if txt:
            return txt
    return None


def webster_meaning(word: str) -> Optional[str]:
    """
    先查原词；如果是英式拼写（里边有 "our"），
    再尝试把 "our" → "or" 转成美式去查一次。
    """
    # 先试原词
    meaning = _webster_meaning_core(word)
    if meaning:
        return meaning

    # 简单英式→美式：neighbourhood → neighborhood, colour → color ...
    if "'" not in word and "our" in word:
        alt = word.replace("our", "or")
        if alt != word:
            alt_meaning = _webster_meaning_core(alt)
            if alt_meaning:
                return alt_meaning

    return None


# ----------------------------
# 4) Cambridge example
# ----------------------------
def cambridge_example(word: str) -> Optional[str]:
    """Return a single example sentence from Cambridge Dictionary."""
    urls = [
        f"https://dictionary.cambridge.org/dictionary/english/{quote(word)}",
        f"https://dictionary.cambridge.org/us/dictionary/english/{quote(word)}",
    ]

    for url in urls:
        print(f"Fetching example for '{word}' from Cambridge...")
        tree = get_tree(url)
        if tree is None:
            continue

        xpaths = [
            "//div[@class='def-body']//span[@class='eg']",
            "//div[@class='examp dexamp']//span",
            "//span[contains(@class,'eg') and not(contains(@class,'dsense'))]",
            "//div[contains(@class,'example-box')]//span[@class='eg']",
            "//div[contains(@class,'examp')]//span[contains(@class,'eg')]",
        ]

        for xp in xpaths:
            try:
                examples = tree.xpath(xp)
                if examples:
                    for ex in examples:
                        text = "".join(ex.xpath(".//text()"))
                        text = re.sub(r"<.*?>", "", text, flags=re.S)
                        text = re.sub(r"\s+", " ", text).strip()
                        if (
                            len(text) > 20
                            and len(text.split()) >= 4
                            and not any(
                                skip in text.lower()
                                for skip in [
                                    "more examples",
                                    "fewer examples",
                                    "smart vocabulary",
                                    "thesaurus",
                                    "see also",
                                    "compare",
                                    "related to",
                                ]
                            )
                        ):
                            print(f"  ✓ Example: {text[:80]}...")
                            return text
            except Exception as e:
                print(f"  XPath error: {e}")
                continue

    return None


# ----------------------------
# 5) Fallback: use article sentence as example
# ----------------------------
def find_sentence_as_example(article_text: str, word: str, max_len: int = 200) -> str:
    """
    从原文中找一条包含该 word 的句子作为例句 fallback。
    简单按 . ? ! 分句，匹配包含该词的片段。
    """
    if not article_text:
        return ""

    pattern = re.compile(
        r"[^.?!]*\b" + re.escape(word) + r"\b[^.?!]*[.?!]",
        flags=re.IGNORECASE,
    )
    m = pattern.search(article_text)
    if not m:
        return ""

    sent = m.group(0)
    # 把 \t / \n 等所有空白合成普通空格
    sent = re.sub(r"\s+", " ", sent).strip()
    if len(sent) > max_len:
        return sent[: max_len - 3].rstrip() + "..."
    return sent


# ----------------------------
# 6) Pipeline helpers
# ----------------------------
def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def is_good_candidate(word: str) -> bool:
    """
    粗暴过滤不想要的词：
      - 包含 ' 的直接丢弃（如 xuanzang's, elephant's）
      - '-' 数量 > 2 的丢弃（如 shwo-yih-tsai-yu-po, minerals--gold 这种奇怪格式）
    """
    if "'" in word:
        return False
    if word.count("-") >= 2:
        return False
    return True


# ----------------------------
# 7) Main build function
# ----------------------------
def build_vocab(article_path: str, select_path: str,
                out_words: str, out_csv: str) -> None:
    # 先把整篇文章读成一个字符串，用于后面 fallback 例句
    try:
        with open(article_path, "r", encoding="utf-8", errors="ignore") as fp:
            article_text = fp.read()
    except FileNotFoundError:
        article_text = ""
        print(f"WARNING: article file {article_path} not found when reading full text.")
    except Exception as e:
        article_text = ""
        print(f"WARNING: failed to read full article text: {e}")

    # 原文 token（保顺序），和 AI / 手写选词表 token
    article_tokens = tokenize_file(article_path)
    selection_tokens = tokenize_file(select_path)

    # 应用“笨方法”过滤 selection 里的词
    filtered_selection = [w for w in selection_tokens if is_good_candidate(w)]
    selected = set(filtered_selection)

    # 只保留：既在原文出现，又在 selection 中，且通过过滤的词；保持原文第一次出现的顺序
    ordered = [w for w in article_tokens if w in selected]
    ordered = unique_preserve_order(ordered)

    # publish progress totals for frontend polling
    _set_progress(status="running", current=0, total=len(ordered), word="")

    print(f"Processing {len(ordered)} words...\n")

    # 写 satword.txt
    with open(out_words, "w", encoding="utf-8") as f:
        for w in ordered:
            f.write(f"{w}\n")

    rows = []
    for i, w in enumerate(ordered, 1):
        _set_progress(status="running", current=i-1, total=len(ordered), word=w)
        print(f"\n[{i}/{len(ordered)}] Processing '{w}'...")

        # 1) Merriam-Webster 释义（带英式→美式 fallback）
        meaning = webster_meaning(w)
        if meaning:
            print(f"  ✓ Meaning: {meaning[:80]}...")
        else:
            print(f"  ✗ No meaning found for '{w}'")

        # 对同一个词，查一次就够，别太快
        time.sleep(0.4)

        # 2) Cambridge 例句，如果没有再从原文里找句子
        example = cambridge_example(w)
        if not example:
            fallback = find_sentence_as_example(article_text, w)
            if fallback:
                print(f"  ✓ Fallback example from article: {fallback[:80]}...")
                example = fallback
            else:
                print(f"  ✗ No example found for '{w}' (including article context)")

        # 3) 如果没有正式释义，则给兜底文案
        if not meaning:
            if example:
                meaning = (
                    "No standard dictionary definition found; "
                    "likely a proper noun, name, or rare term. "
                    "See the example sentence for context."
                )
            else:
                meaning = (
                    "No standard dictionary definition found; "
                    "likely a proper noun, name, or rare term. "
                    "See the article for context."
                )

        rows.append([w, meaning or "", example or ""])

        # 随机 sleep 一下，避免被网站当成机器人
        time.sleep(random.uniform(0.5, 1.5))

        # mark this word as completed
        _set_progress(status="running", current=i, total=len(ordered), word=w)

    # 写 CSV
    try:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "meaning", "example"])
            writer.writerows(rows)
        print(f"\n✓ Success! Wrote {out_csv}")
        _set_progress(status="done", current=len(ordered), total=len(ordered), word="")
    except PermissionError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem, dot, ext = out_csv.partition(".")
        alt = f"{stem}_{ts}.{ext}" if dot else f"{out_csv}_{ts}"
        with open(alt, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "meaning", "example"])
            writer.writerows(rows)
        print(f"WARNING: Could not write {out_csv} (file in use). Wrote {alt} instead.")
        _set_progress(status="done", current=len(ordered), total=len(ordered), word="")


# ----------------------------
# 8) CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build vocabulary CSV from article + selection list."
    )
    ap.add_argument(
        "--article",
        default="sat.txt",
        help="Path to the article .txt (default: sat.txt)",
    )
    ap.add_argument(
        "--select",
        default="20241226.txt",
        help="Path to the selection list .txt (default: 20241226.txt)",
    )
    ap.add_argument(
        "--out_words",
        default="satword.txt",
        help="Output words list (default: satword.txt)",
    )
    ap.add_argument(
        "--out_csv",
        default="words.csv",
        help="Output CSV (default: words.csv)",
    )
    args = ap.parse_args()

    build_vocab(args.article, args.select, args.out_words, args.out_csv)
    print(f"Done. Wrote {args.out_words} and {args.out_csv}")


if __name__ == "__main__":
    main()
