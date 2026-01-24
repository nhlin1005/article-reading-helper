import os
import sys
import json
from flask import Flask, request, jsonify, send_from_directory

# ------------- 路径设置：把 article-reading-helper 加到 Python 搜索路径 -------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELPER_DIR = os.path.join(BASE_DIR, "article-reading-helper")
sys.path.append(HELPER_DIR)

# 从你的 pipeline_from_pdf.py 里导入 run_ai_mode
from pipeline_from_pdf import run_ai_mode   # 如果函数名不同，请对照文件改这里

# ------------- Flask 初始化，顺便当静态服务器用（直接跑前端） -------------
app = Flask(
    __name__,
    static_folder=BASE_DIR,      # 把整个根目录当静态文件目录
    static_url_path=""           # 这样 /index.html、/spartan-logo.png 都能直接访问
)

# 首页：返回 index.html
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# 兜底静态资源，比如 spartan-logo.png、README 等
@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(BASE_DIR, path)


# ------------- 核心接口：接收 PDF，调用 article-reading-helper，回前端 JSON -------------
@app.post("/api/extract_keywords")
def extract_keywords():
    """
    前端传一个 pdf 文件字段名叫 'pdf'
    -> 保存到 article-reading-helper/data/
    -> 调 run_ai_mode(...)
    -> 读 reading_xxx/xxx.ai.json
    -> 返回 {words: [...], wordData: {...}} 给前端
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No file field 'pdf' found"}), 400

    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 安全的文件名和 stem
    original_name = pdf_file.filename
    name_no_ext, _ = os.path.splitext(original_name)
    safe_stem = name_no_ext.replace(" ", "_").replace("/", "_")

    # 保存 PDF 到 article-reading-helper/data
    data_dir = os.path.join(HELPER_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    pdf_path = os.path.join(data_dir, original_name)
    pdf_file.save(pdf_path)

    # 设定 reading 结果目录（和你命令行生成的一样风格）
    reading_dir = os.path.join(HELPER_DIR, f"reading_{safe_stem}")
    os.makedirs(reading_dir, exist_ok=True)

    # 调用你原来的流水线：提取文本 + 选词 + 词典 + 生成 JSON
    # ai_top_n 自己调，可以不写让它用默认值，也可以固定 80、50 等
    ai_top_n = 80
    try:
        run_ai_mode(
            pdf_path=pdf_path,
            reading_dir=reading_dir,
            safe_stem=safe_stem,
            ai_top_n=ai_top_n,
        )
    except TypeError:
        # 如果你后来改了函数签名，这里可以退回最简单版本
        # 例如：run_ai_mode(pdf_path, reading_dir, safe_stem)
        run_ai_mode(pdf_path, reading_dir, safe_stem)

    # 读取生成好的 JSON，比如 reading_Xuanzang-page_1-5/Xuanzang-page_1-5.ai.json
    json_path = os.path.join(reading_dir, f"{safe_stem}.ai.json")
    if not os.path.exists(json_path):
        return jsonify({"error": f"JSON vocab file not found: {json_path}"}), 500

    with open(json_path, "r", encoding="utf-8") as f:
        vocab_list = json.load(f)

    # 转成前端好用的结构：words + wordData
    words = []
    wordData = {}
    for item in vocab_list:
        # 处理 "﻿word" 这种带 BOM 的 key
        w = (
            item.get("word")
            or item.get("\ufeffword")
            or item.get("﻿word")  # 某些编辑器会显示成这个
        )
        if not w:
            continue

        words.append(w)
        wordData[w] = {
            "meaning": item.get("meaning", ""),
            "example": item.get("example", ""),
        }

    return jsonify({"words": words, "wordData": wordData})


if __name__ == "__main__":
    # 在 word check\word check 目录下运行：
    #   python server.py
    app.run(host="127.0.0.1", port=5000, debug=True)
