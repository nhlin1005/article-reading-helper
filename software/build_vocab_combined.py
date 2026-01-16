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
  - words.csv          (word, meaning-from-Webster, example-from-Cambridge)

Rule:
  - MEANING from Merriam-Webster
  - EXAMPLE from Cambridge Dictionary (more reliable for automated access)
  - No 'source' column in CSV

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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
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


def clean_text_list(nodes: Iterable[str]) -> List[str]:
    """Remove tags, collapse whitespace, strip punctuation artifacts."""
    clean = []
    for n in nodes:
        n = re.sub(r"<.*?>", "", n, flags=re.S)
        n = re.sub(r"\s+", " ", n).strip(" :;\n\t\r")
        if n:
            clean.append(n)
    return clean


# ----------------------------
# 3) Data sources per new rule
# ----------------------------
def webster_meaning(word: str) -> Optional[str]:
    """Return a short meaning from Merriam-Webster (first sense)."""
    url = f"https://www.merriam-webster.com/dictionary/{word}"
    tree = get_tree(url)
    if tree is None:
        return None

    # Definitions
    def_nodes = tree.xpath(
        "//div[contains(@id,'dictionary-entry')]/div[@class='vg']//span[@class='dtText' or @class='unText']")
    for n in def_nodes:
        txt = html.tostring(n, encoding='unicode')
        txt = re.sub(r'<.*?>', '', txt)
        txt = re.sub(r'^\s*:\s*', '', txt).strip()
        if txt:
            return txt
    return None


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

        # Updated XPath selectors for current website structure
        xpaths = [
            # Primary example sentences
            "//div[@class='def-body']//span[@class='eg']",
            "//div[@class='examp dexamp']//span",
            "//span[contains(@class,'eg') and not(contains(@class,'dsense'))]",
            # Alternative structures
            "//div[contains(@class,'example-box')]//span[@class='eg']",
            "//div[contains(@class,'examp')]//span[contains(@class,'eg')]",
        ]

        for xp in xpaths:
            try:
                examples = tree.xpath(xp)
                if examples:
                    for ex in examples:
                        text = ''.join(ex.xpath('.//text()'))
                        text = re.sub(r"<.*?>", "", text, flags=re.S)
                        text = re.sub(r"\s+", " ", text).strip()
                        # More strict filtering for quality examples
                        if (len(text) > 20 and
                                len(text.split()) >= 4 and
                                not any(skip in text.lower() for skip in
                                        ['more examples', 'fewer examples', 'smart vocabulary',
                                         'thesaurus', 'see also', 'compare', 'related to'])):
                            print(f"  ✓ Example: {text[:80]}...")
                            return text
            except Exception as e:
                print(f"  XPath error: {e}")
                continue

    return None


# ----------------------------
# 4) Pipeline
# ----------------------------
def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_vocab(article_path: str, select_path: str,
                out_words: str, out_csv: str) -> None:
    # tokens from files (lowercased/normalized)
    article_tokens = tokenize_file(article_path)
    selection_tokens = tokenize_file(select_path)

    # intersection, preserving order of first appearance in article
    selected = set(selection_tokens)
    ordered = [w for w in article_tokens if w in selected]
    ordered = unique_preserve_order(ordered)

    print(f"Processing {len(ordered)} words...\n")

    # write satword.txt
    with open(out_words, "w", encoding="utf-8") as f:
        for w in ordered:
            f.write(f"{w}\n")

    # look up rows first
    rows = []
    for i, w in enumerate(ordered, 1):
        print(f"\n[{i}/{len(ordered)}] Processing '{w}'...")

        # Get meaning from Merriam-Webster
        meaning = webster_meaning(w)
        if meaning:
            print(f"  ✓ Meaning: {meaning[:80]}...")
        else:
            print(f"  ✗ No meaning found for '{w}'")

        # Add a small delay between requests
        time.sleep(0.4)

        # Get example from Cambridge
        example = cambridge_example(w)
        if not example:
            print(f"  ✗ No example found for '{w}'")

        rows.append([w, meaning or "", example or ""])

        # Random delay between requests
        time.sleep(random.uniform(0.5, 1.5))

    # write csv (no source column), with fallback if target is locked
    try:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "meaning", "example"])
            writer.writerows(rows)
        print(f"\n✓ Success! Wrote {out_csv}")
    except PermissionError:
        import os
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem, dot, ext = out_csv.partition(".")
        alt = f"{stem}_{ts}.{ext}" if dot else f"{out_csv}_{ts}"
        with open(alt, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "meaning", "example"])
            writer.writerows(rows)
        print(f"WARNING: Could not write {out_csv} (file in use). Wrote {alt} instead.")


# ----------------------------
# 5) CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build vocabulary CSV from article + selection list.")
    ap.add_argument("--article", default="sat.txt", help="Path to the article .txt (default: sat.txt)")
    ap.add_argument("--select", default="20241226.txt", help="Path to the selection list .txt (default: 20241226.txt)")
    ap.add_argument("--out_words", default="satword.txt", help="Output words list (default: satword.txt)")
    ap.add_argument("--out_csv", default="words.csv", help="Output CSV (default: words.csv)")
    args = ap.parse_args()

    build_vocab(args.article, args.select, args.out_words, args.out_csv)
    print(f"Done. Wrote {args.out_words} and {args.out_csv}")


if __name__ == "__main__":
    main()
