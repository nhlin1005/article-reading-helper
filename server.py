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
BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "Frontend"
AIMODE_DIR   = BASE_DIR / "aimode"
DATA_DIR     = AIMODE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

XLSX_PATH  = BASE_DIR / "words_with_examples.xlsx"   # main vocabulary dictionary
XLSX_SHEET = "All_Words"                              # sheet name inside the XLSX
CACHE_DB   = str(BASE_DIR / "vocab_cache.sqlite")    # shared SQLite cache

# Allow bare imports from aimode/ and project root
sys.path.insert(0, str(AIMODE_DIR))
sys.path.insert(0, str(BASE_DIR))

from pipeline_from_pdf import run_ai_mode
from build_vocab_combined import build_vocab
import build_vocab_combined as bvc

# Standalone single-word lookup: XLSX → SQLite cache → web scrape
# No article context needed — works any time.
from lookup_single_word import lookup_word_in_xlsx

# ---------- Global article context (set after AI Extract) ----------
CURRENT_CONTEXT = {
    "reading_dir": None,
    "safe_stem":   None,
    "article_txt": None,
    "score_json":  None,
}

# ---------- Flask app ----------
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)


# ── Progress ─────────────────────────────────────────────────────────────────
@app.route("/api/progress", methods=["GET"])
def get_progress():
    try:
        prog = getattr(bvc, "PROGRESS", None) or {}
        return jsonify({
            "status":  prog.get("status",  "idle"),
            "current": int(prog.get("current", 0) or 0),
            "total":   int(prog.get("total",   0) or 0),
            "word":    prog.get("word", "") or "",
        })
    except Exception:
        return jsonify({"status": "idle", "current": 0, "total": 0, "word": ""})


# ── Static frontend ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _try_find_score_json(reading_dir: Path, safe_stem: str) -> str:
    candidates = [
        reading_dir / "selected_words.json",
        reading_dir / f"{safe_stem}.selected_words.json",
        reading_dir / f"{safe_stem}.scores.json",
        reading_dir / f"{safe_stem}.selected.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
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


# ── AI Extract (full pipeline) ────────────────────────────────────────────────
@app.route("/api/extract_keywords", methods=["POST"])
def extract_keywords():
    if "pdf" not in request.files:
        return jsonify({"error": "No 'pdf' file in request"}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        ai_top_n = float(request.form.get("ai_top_n", "0.1"))
    except ValueError:
        ai_top_n = 0.1

    safe_name = secure_filename(pdf_file.filename)
    pdf_path  = DATA_DIR / safe_name
    pdf_file.save(pdf_path)

    try:
        if hasattr(bvc, "PROGRESS") and isinstance(bvc.PROGRESS, dict):
            bvc.PROGRESS.update({"status": "running", "current": 0, "total": 0, "word": ""})
    except Exception:
        pass

    stem        = pdf_path.stem
    safe_stem   = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    reading_dir = BASE_DIR / f"reading_{safe_stem}"
    reading_dir.mkdir(parents=True, exist_ok=True)

    run_ai_mode(pdf_path, reading_dir, safe_stem, ai_top_n=ai_top_n)

    json_path = reading_dir / f"{safe_stem}.ai.json"
    if not json_path.exists():
        return jsonify({"error": f"Result JSON not found: {json_path}"}), 500

    with json_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    CURRENT_CONTEXT["reading_dir"] = reading_dir
    CURRENT_CONTEXT["safe_stem"]   = safe_stem
    CURRENT_CONTEXT["article_txt"] = reading_dir / f"{safe_stem}.txt"
    score_json_path = _try_find_score_json(reading_dir, safe_stem)
    CURRENT_CONTEXT["score_json"]  = score_json_path or None

    words     = []
    word_data = {}
    for e in entries:
        w = e.get("word") or e.get("\ufeffword") or e.get("Word")
        if not w:
            continue
        words.append(w)
        word_data[w] = {
            "meaning": e.get("meaning", ""),
            "example": e.get("example", ""),
        }

    words = sorted(set(words), key=lambda s: (s or "").lower())
    return jsonify({
        "words":         words,
        "wordData":      word_data,
        "readingFolder": f"reading_{safe_stem}",
        "jsonFile":      json_path.name,
        "scoreJson":     CURRENT_CONTEXT["score_json"] or "",
    })


# ── Single-word lookup ────────────────────────────────────────────────────────
@app.route("/api/lookup_word", methods=["POST"])
def lookup_word():
    """
    Look up definition + example for one word.

    Priority:
      1. lookup_single_word.py:
           a. XLSX dictionary  (same column logic as build_vocab_combined)
           b. SQLite cache     (shared with build_vocab_combined)
           c. Web scrape       (Merriam-Webster + Cambridge)
         → works without any article context, always available.

      2. build_vocab pipeline fallback (only if step 1 returns nothing AND
         an article context exists from a prior AI Extract run).
         Now passes the correct xlsx + sheet + cache args.
    """
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    if not word:
        return jsonify({"error": "Missing 'word'"}), 400

    # ── 1. XLSX / cache / web-scrape ─────────────────────────────────────────
    result = lookup_word_in_xlsx(word, str(XLSX_PATH), cache_db=CACHE_DB)
    if result.get("meaning") or result.get("example"):
        return jsonify({
            "word":    word,
            "meaning": result.get("meaning", ""),
            "example": result.get("example", ""),
            "ipa":     result.get("ipa", ""),
        })

    # ── 2. build_vocab pipeline fallback ────────────────────────────────────
    ctx         = CURRENT_CONTEXT
    reading_dir = ctx.get("reading_dir")
    article_txt = ctx.get("article_txt")

    if reading_dir and article_txt and Path(article_txt).exists():
        reading_dir = Path(reading_dir)
        safe_word   = "".join(c if (c.isalnum() or c in "-_") else "_" for c in word)
        select_path = reading_dir / f"_tmp_select_{safe_word}.txt"
        out_words   = reading_dir / f"_tmp_words_{safe_word}.txt"
        out_csv     = reading_dir / f"_tmp_vocab_{safe_word}.csv"
        select_path.write_text(word + "\n", encoding="utf-8")

        meaning = ""
        example = ""
        try:
            # Pass xlsx + sheet + cache — these were missing before and caused
            # build_vocab to skip the dictionary and fail silently.
            build_vocab(
                str(article_txt),
                str(select_path),
                str(out_words),
                str(out_csv),
                str(XLSX_PATH),   # lookup_xlsx
                XLSX_SHEET,       # xlsx_sheet
                cache_db=CACHE_DB,
                workers=1,
            )

            if out_csv.exists():
                with out_csv.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        wk = next((k for k in row if k.strip("\ufeff").lower() == "word"), None)
                        if wk and (row.get(wk) or "").strip().lower() == word.lower():
                            meaning = (row.get("meaning") or "").strip()
                            example = (row.get("example") or "").strip()
                            break
        except Exception as e:
            print(f"[lookup_word] build_vocab fallback failed for '{word}': {e}")
        finally:
            for p in (select_path, out_words, out_csv):
                try: p.unlink()
                except Exception: pass

        if meaning or example:
            return jsonify({"word": word, "meaning": meaning, "example": example, "ipa": ""})

    # ── Nothing found ─────────────────────────────────────────────────────────
    return jsonify({
        "word":    word,
        "meaning": "No definition available.",
        "example": "No example available.",
        "ipa":     "",
    })


# ── Article text (for TTS) ────────────────────────────────────────────────────
@app.route("/api/article_text", methods=["GET"])
def api_article_text():
    try:
        txt = CURRENT_CONTEXT.get("article_txt")
        if not txt:
            return jsonify({"text": ""})
        p = Path(txt)
        return jsonify({"text": p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""})
    except Exception:
        return jsonify({"text": ""})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)