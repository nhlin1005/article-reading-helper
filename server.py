# server.py
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import sys
import json
import os

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent          # project root
FRONTEND_DIR = BASE_DIR / "Frontend"                # index.html, index.js
AIMODE_DIR = BASE_DIR / "aimode"                    # all backend NLP code
DATA_DIR = AIMODE_DIR / "data"                      # where uploaded PDFs go
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Let Python import modules inside aimode/ by bare name:
#   from pipeline_from_pdf import run_ai_mode
sys.path.insert(0, str(AIMODE_DIR))

from pipeline_from_pdf import run_ai_mode           # uses ai_select_wordlist, build_vocab, etc.

# ---------- Flask app ----------
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path=""
)

CORS(app)


# ---------- Routes ----------
@app.route("/")
def index():
    """Serve the front-end."""
    return send_from_directory(FRONTEND_DIR, "index.html")


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

    # create reading_{safe_stem} folder under project root (same logic as pipeline_from_pdf)
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

    # convert list[{word, meaning, example}] -> {words: [...], wordData: {...}} for the front-end
    words = []
    word_data = {}

    for e in entries:
        # handle possible BOM on "word" key
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

    return jsonify({
        "words": words,
        "wordData": word_data,
        "readingFolder": f"reading_{safe_stem}",
        "jsonFile": json_path.name,
    })


if __name__ == "__main__":
    # run from project root:
    #   python server.py
    app.run(host="0.0.0.0", port=5000, debug=False)
