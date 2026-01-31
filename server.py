# server.py
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import sys
import json
import os
import csv

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent          # project root
FRONTEND_DIR = BASE_DIR / "Frontend"                # index.html, index.js
AIMODE_DIR = BASE_DIR / "aimode"                    # all backend NLP code
DATA_DIR = AIMODE_DIR / "data"                      # where uploaded PDFs go
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Let Python import modules inside aimode/ by bare name:
sys.path.insert(0, str(AIMODE_DIR))

from pipeline_from_pdf import run_ai_mode           # PDF -> txt -> vocab JSON
from build_vocab_combined import build_vocab        # core lookup pipeline
import build_vocab_combined as bvc

# ---------- Global “current article” context ----------
CURRENT_CONTEXT = {
    "reading_dir": None,   # Path("reading_XXXX")
    "safe_stem": None,     # "Xuanzang-page_1-5"
    "article_txt": None,   # Path to article txt inside reading_dir
    "score_json": None,    # Path to score json (optional)
}

# ---------- Flask app ----------
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)

CORS(app)


@app.route("/api/progress", methods=["GET"])
def get_progress():
    """Expose live progress of the vocabulary build step.

    Returns:
      { status: 'idle'|'running'|'done', current: int, total: int, word: str }
    """
    try:
        prog = getattr(bvc, "PROGRESS", None) or {}
        return jsonify({
            "status": prog.get("status", "idle"),
            "current": int(prog.get("current", 0) or 0),
            "total": int(prog.get("total", 0) or 0),
            "word": prog.get("word", "") or "",
        })
    except Exception:
        return jsonify({"status": "idle", "current": 0, "total": 0, "word": ""})


# ---------- Routes ----------
@app.route("/")
def index():
    """Serve the front-end."""
    return send_from_directory(FRONTEND_DIR, "index.html")


def _try_find_score_json(reading_dir: Path, safe_stem: str) -> str:
    """
    试着在 reading_dir 里找 AI 选词阶段生成的 score json（文件名不确定，所以做自动探测）。
    找到就返回路径字符串，否则返回 ""。
    """
    candidates = [
        reading_dir / "selected_words.json",
        reading_dir / f"{safe_stem}.selected_words.json",
        reading_dir / f"{safe_stem}.scores.json",
        reading_dir / f"{safe_stem}.selected.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # 再兜底：找最像的 json（但避免拿到 ai.json）
    try:
        for p in reading_dir.glob("*.json"):
            name = p.name.lower()
            if name.endswith(".ai.json"):
                continue
            if "score" in name or "select" in name or "word" in name:
                return str(p)
    except Exception:
        pass

    return ""


@app.route("/api/extract_keywords", methods=["POST"])
def extract_keywords():
    """
    API called by the front-end.

    Expected form fields:
      - pdf: the uploaded PDF file
      - ai_top_n: optional, float/int; >=1 = number of words, 0~1 = ratio (e.g. 0.1 = 10%)
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No 'pdf' file in request"}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # parse ai_top_n (default 0.1 = 10% of candidates)
    ai_top_n_str = request.form.get("ai_top_n", "0.1")
    try:
        ai_top_n = float(ai_top_n_str)
    except ValueError:
        ai_top_n = 0.1

    # save uploaded PDF into aimode/data/
    safe_name = secure_filename(pdf_file.filename)
    pdf_path = DATA_DIR / safe_name
    pdf_file.save(pdf_path)

    # reset live progress (so frontend starts from 0/?)
    try:
        if hasattr(bvc, "PROGRESS") and isinstance(bvc.PROGRESS, dict):
            bvc.PROGRESS.update({"status": "running", "current": 0, "total": 0, "word": ""})
    except Exception:
        pass

    # create reading_{safe_stem} folder under project root
    stem = pdf_path.stem
    safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    reading_dir = BASE_DIR / f"reading_{safe_stem}"
    reading_dir.mkdir(parents=True, exist_ok=True)

    # run the full AI pipeline (PDF -> txt -> AI select -> CSV -> JSON + refine)
    run_ai_mode(pdf_path, reading_dir, safe_stem, ai_top_n=ai_top_n)

    # the refined JSON lives at: reading_{safe_stem}/{safe_stem}.ai.json
    json_path = reading_dir / f"{safe_stem}.ai.json"
    if not json_path.exists():
        return jsonify({"error": f"Result JSON not found: {json_path}"}), 500

    with json_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    # remember current context so we can look up extra words later
    CURRENT_CONTEXT["reading_dir"] = reading_dir
    CURRENT_CONTEXT["safe_stem"] = safe_stem
    CURRENT_CONTEXT["article_txt"] = reading_dir / f"{safe_stem}.txt"

    # [新增] 尝试记录 score_json（如果 pipeline 生成了的话）
    score_json_path = _try_find_score_json(reading_dir, safe_stem)
    CURRENT_CONTEXT["score_json"] = score_json_path if score_json_path else None

    # convert list[{word, meaning, example}] -> {words: [...], wordData: {...}} for the front-end
    words = []
    word_data = {}

    for e in entries:
        w = e.get("word") or e.get("﻿word") or e.get("Word")
        if not w:
            continue

        meaning = e.get("meaning", "")
        example = e.get("example", "")
        words.append(w)
        word_data[w] = {
            "meaning": meaning,
            "example": example,
        }

    words = sorted(set(words), key=lambda s: (s or '').lower())
    return jsonify({
        "words": words,
        "wordData": word_data,
        "readingFolder": f"reading_{safe_stem}",
        "jsonFile": json_path.name,
        "scoreJson": CURRENT_CONTEXT["score_json"] or "",  # 给前端可选展示（不影响）
    })


@app.route("/api/lookup_word", methods=["POST"])
def lookup_word():
    """
    For a single word, reuse build_vocab_combined to fetch meaning/example.

    Expected JSON body:
      { "word": "addressing" }

    Returns:
      { "word": ..., "meaning": ..., "example": ... }
    """
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        return jsonify({"error": "Missing 'word'"}), 400

    ctx = CURRENT_CONTEXT
    reading_dir = ctx.get("reading_dir")
    article_txt = ctx.get("article_txt")
    score_json = ctx.get("score_json") or ""  # optional

    if not reading_dir or not article_txt or not Path(article_txt).exists():
        return jsonify({
            "error": "No article context yet. Please upload and process a PDF first."
        }), 400

    reading_dir = Path(reading_dir)
    article_txt = Path(article_txt)

    # temp files inside the same reading_dir
    safe_word = "".join(c if (c.isalnum() or c in "-_") else "_" for c in word)
    select_path = reading_dir / f"_tmp_select_{safe_word}.txt"
    out_words = reading_dir / f"_tmp_words_{safe_word}.txt"
    out_csv = reading_dir / f"_tmp_vocab_{safe_word}.csv"

    # write select list (single word)
    select_path.write_text(word + "\n", encoding="utf-8")

    # run the same pipeline just for this one word
    try:
        # ✅ 兼容你“新 build_vocab 签名(带 score_json_path)”和旧签名
        try:
            build_vocab(str(article_txt), str(select_path), str(out_words), str(out_csv), score_json)
        except TypeError:
            # 旧版 build_vocab 只有 4 个参数
            build_vocab(str(article_txt), str(select_path), str(out_words), str(out_csv))
    except Exception as e:
        for p in (select_path, out_words, out_csv):
            try:
                p.unlink()
            except Exception:
                pass
        return jsonify({
            "word": word,
            "meaning": "No definition available.",
            "example": "No example available.",
            "error": f"lookup failed: {e}",
        })

    # parse CSV result
    meaning = ""
    example = ""
    if out_csv.exists():
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                word_key = None
                for k in row.keys():
                    if k.strip("\ufeff").lower() == "word":
                        word_key = k
                        break
                if not word_key:
                    continue

                w_val = (row.get(word_key) or "").strip()
                if w_val.lower() == word.lower():
                    meaning = (row.get("meaning") or "").strip()
                    example = (row.get("example") or "").strip()
                    break

    # cleanup temp files
    for p in (select_path, out_words, out_csv):
        try:
            p.unlink()
        except Exception:
            pass

    if not meaning:
        meaning = "No definition available."
    if not example:
        example = "No example available."

    return jsonify({
        "word": word,
        "meaning": meaning,
        "example": example,
    })


if __name__ == "__main__":
    # run from project root:
    #   python server.py
    # threaded=True is required so the frontend can poll /api/progress
    # while the long /api/extract_keywords request is still running.
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
