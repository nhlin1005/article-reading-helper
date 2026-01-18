# -*- coding: utf-8 -*-
"""
从 PDF 一条龙跑完选词流程，并把所有结果放进一个文件夹：
  reading_{文章名}

支持两种模式：
  1) --mode ai
       用 AI 自动选词（不需要手工 select 词表）
  2) --mode list
       用你自己准备好的 select 词表（和之前一样）

用法示例：

  # 1. AI 模式（推荐）
  python pipeline_from_pdf.py \
      --mode ai \
      --pdf "data/Xuanzang-page 1-5.pdf" \
      --ai_top_n 30

  # 2. 手工列表模式
  python pipeline_from_pdf.py \
      --mode list \
      --pdf "data/Xuanzang-page 1-5.pdf" \
      --select 20241226.txt
"""

import argparse
from pathlib import Path

from extract_pdf_text import extract_text_from_pdf
from build_vocab_combined import build_vocab
from csv_to_json import csv_to_json
from ai_select_wordlist import ai_select_words_for_article
from config import DEFAULT_TOP_N


def _make_reading_folder(pdf_path: Path):
    """
    根据 PDF 文件生成：
      - safe_stem: 去掉空格等的安全文件名
      - reading_dir: reading_{safe_stem} 文件夹路径
      - article_txt_path: 文章 txt 路径（放在 reading_dir 里）
    """
    pdf_stem = pdf_path.stem  # 原始文件名（不含后缀），如 "Xuanzang-page 1-5"

    # 把空格等转成下划线，避免路径太怪
    safe_stem = "".join(
        c if (c.isalnum() or c in "-_") else "_" for c in pdf_stem
    )

    reading_dir = Path(f"reading_{safe_stem}")
    reading_dir.mkdir(parents=True, exist_ok=True)

    article_txt_path = reading_dir / f"{safe_stem}.txt"
    return reading_dir, safe_stem, article_txt_path


def run_ai_mode(pdf_path: Path,
                reading_dir: Path,
                safe_stem: str,
                ai_top_n: int = DEFAULT_TOP_N):
    """
    使用 AI 模式：
      PDF -> txt -> AI 选词表 -> words.txt + csv + json
    全部输出到 reading_dir 下面。
    """
    # 1) PDF -> TXT
    article_txt = reading_dir / f"{safe_stem}.txt"
    print(f"\n[Step 1] 从 PDF 提取文本：{pdf_path} -> {article_txt}")
    ok = extract_text_from_pdf(str(pdf_path), str(article_txt))
    if not ok:
        raise SystemExit("❌ PDF 文本提取失败，退出。")

    text = article_txt.read_text(encoding="utf-8", errors="ignore")

    # 2) AI 选词（生成一个等价于 select_XXXX.txt 的词表）
    select_path = reading_dir / f"{safe_stem}.ai.select.txt"
    print(f"\n[Step 2] 用 AI 从文章生成生词表（最多 {ai_top_n} 个词）...")
    words = ai_select_words_for_article(text, top_n=ai_top_n)
    select_path.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"  ✓ AI 选词表已保存：{select_path} （共 {len(words)} 个词）")

    # 3) build_vocab：文章 txt + AI 词表 -> words.txt + csv
    out_words = reading_dir / f"{safe_stem}.ai.words.txt"
    out_csv = reading_dir / f"{safe_stem}.ai.csv"

    print(f"\n[Step 3] 构建词汇表 CSV（查释义 + 例句）...")
    # ⚠️ 这里改成位置参数，和 build_vocab_combined.py 里的签名一致
    build_vocab(str(article_txt), str(select_path), str(out_words), str(out_csv))
    print(f"  ✓ 文章中出现的目标词列表：{out_words}")
    print(f"  ✓ 词汇 CSV：{out_csv}")

    # 4) CSV -> JSON
    out_json = reading_dir / f"{safe_stem}.ai.json"
    print(f"\n[Step 4] 把 CSV 转成 JSON：{out_csv} -> {out_json}")
    csv_to_json(str(out_csv), str(out_json))
    print(f"  ✓ 词汇 JSON：{out_json}")

    print("\n✅ AI 模式完整结束，所有文件都在：", reading_dir)


def run_list_mode(pdf_path: Path,
                  reading_dir: Path,
                  safe_stem: str,
                  select_path: Path):
    """
    使用手工列表模式：
      PDF -> txt
      txt + 用户提供的 select.txt -> words.txt + csv + json
    全部输出到 reading_dir 下面。
    """
    if not select_path.exists():
        raise SystemExit(f"❌ 提供的 select 词表不存在：{select_path}")

    # 1) PDF -> TXT
    article_txt = reading_dir / f"{safe_stem}.txt"
    print(f"\n[Step 1] 从 PDF 提取文本：{pdf_path} -> {article_txt}")
    ok = extract_text_from_pdf(str(pdf_path), str(article_txt))
    if not ok:
        raise SystemExit("❌ PDF 文本提取失败，退出。")

    # 2) build_vocab：文章 txt + 用户词表 -> words.txt + csv
    out_words = reading_dir / f"{safe_stem}.list.words.txt"
    out_csv = reading_dir / f"{safe_stem}.list.csv"

    print(f"\n[Step 2] 使用你提供的词表构建 CSV...")
    # ⚠️ 同样改成位置参数
    build_vocab(str(article_txt), str(select_path), str(out_words), str(out_csv))
    print(f"  ✓ 文章中出现的目标词列表：{out_words}")
    print(f"  ✓ 词汇 CSV：{out_csv}")

    # 3) CSV -> JSON
    out_json = reading_dir / f"{safe_stem}.list.json"
    print(f"\n[Step 3] 把 CSV 转成 JSON：{out_csv} -> {out_json}")
    csv_to_json(str(out_csv), str(out_json))
    print(f"  ✓ 词汇 JSON：{out_json}")

    print("\n✅ 手工列表模式完整结束，所有文件都在：", reading_dir)


def main():
    ap = argparse.ArgumentParser(
        description="从 PDF 生成阅读用的生词表 & CSV & JSON（支持 AI 选词 or 手工 select）"
    )
    ap.add_argument(
        "--mode",
        choices=["ai", "list"],
        required=True,
        help="ai = 使用 AI 自动选词; list = 使用你提供的 select 词表",
    )
    ap.add_argument("--pdf", required=True, help="输入 PDF 文件路径")
    ap.add_argument(
        "--select",
        help="当 --mode list 时，指定你的词表 txt（类似 20241226.txt）",
    )
    ap.add_argument(
        "--ai_top_n",
        type=float,
        default=DEFAULT_TOP_N,
        help=(
            "AI 模式时选多少个词："
            ">=1 表示具体数量（例如 30），"
            "0~1 表示比例（例如 0.1 = 候选词的 10%）"
        ),
    )

    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"❌ PDF 文件不存在：{pdf_path}")

    reading_dir, safe_stem, article_txt_path = _make_reading_folder(pdf_path)
    print(f"📁 本次输出文件夹：{reading_dir}")
    print(f"📝 文章 txt 将保存为：{article_txt_path.name}")

    if args.mode == "ai":
        run_ai_mode(pdf_path, reading_dir, safe_stem, ai_top_n=args.ai_top_n)
    else:
        if not args.select:
            raise SystemExit("❌ --mode list 需要提供 --select <你的词表.txt>")
        run_list_mode(pdf_path, reading_dir, safe_stem, Path(args.select))


if __name__ == "__main__":
    main()
