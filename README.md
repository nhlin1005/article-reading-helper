# article-reading-helper

An online article reading helper that turns **PDF articles into study-ready word lists**.  
It supports both **manual word lists** and an **AI model that automatically picks difficult words**.

> 原始目标：给学生一篇英文文章，自动抽取“可能不认识的单词”，生成带释义和例句的生词本，并且可以以后接到网页 / App 上使用。

---

## 1. 项目结构 (Project Layout)

```text
article-reading-helper/
├── data/                         # 你的 PDF / 文章数据放这里（示例：Xuanzang-page 1-5.pdf）
├── extract_pdf_text.py           # PDF → TXT 的小工具（原始版本）
├── csv_to_json.py                # 把 CSV 词表转成 JSON
├── build_vocab_combined.py       # 传统模式：用“给定单词表”+ 文章生成词汇表
├── train_keyword_model.py        # 训练 AI 关键词抽取模型（Inspec 数据集）
├── keyword_extractor.py          # 用训练好的模型对文本做关键词抽取
├── candidate_builder.py          # 从文章构造候选单词 + 频率统计、过滤规则
├── ai_select_wordlist.py         # 用 AI + 规则，给整篇文章选出生词表
├── text_utils.py                 # 文本清洗、分句等工具函数
├── llm_refiner.py                # （可选）以后可以接大模型进一步 refine 结果
├── pipeline_from_pdf.py          # 从 PDF 一键到：txt + 词表 CSV + JSON（支持 AI / 手动两种模式）
├── keyword-bert-inspec/         # 训练好后的模型目录（训练脚本会自动创建）
│   ├── config.json
│   ├── model.safetensors
│   ├── vocab.txt
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── README.md                     # 本说明文件
```

---

## 2. 环境准备 (Environment)

建议使用 Conda 创建独立环境（Python 3.10）：

```bash
conda create -n article-helper python=3.10
conda activate article-helper
```

安装依赖（示例）：

```bash
pip install torch transformers datasets seqeval requests beautifulsoup4 pandas
```

> 你实验室服务器里的环境已经能成功跑 `train_keyword_model.py` 和 `pipeline_from_pdf.py`，本地只要装好 **同一版本的 `transformers/torch` 和相关依赖** 就能直接加载模型使用。

---

## 3. 原始功能：PDF → TXT + 手动单词表模式

### 3.1 PDF 转 TXT

脚本：`extract_pdf_text.py`  
用途：把 PDF 转成纯文本，便于后续处理。

示例：

```bash
python extract_pdf_text.py --input data/your_article.pdf --output your_article.txt
```

- `--input`：PDF 文件路径  
- `--output`：输出的 txt 文件名

---

### 3.2 传统模式：用给定单词表生成词汇表

脚本：`build_vocab_combined.py`  
用途：用“文章 txt + 自己准备的单词表 txt”生成词汇 CSV 和 words.txt

- **释义**：来自 *Merriam-Webster*  
- **例句**：来自 *Cambridge Dictionary*

示例命令：

```bash
python build_vocab_combined.py \
  --article your_article.txt \
  --select  your_word_list.txt \
  --out_words words.txt \
  --out_csv   words.csv
```

参数说明：

- `--article`：文章的 txt 文件（比如刚从 PDF 转出来的）
- `--select`：你自己准备的生词候选表（每行一个单词）
- `--out_words`：输出：文章里真正出现过的目标词列表（txt）
- `--out_csv`：输出：带释义和例句的词汇表（CSV）

---

## 4. 新功能：AI 关键词模型（自动选词）

### 4.1 训练 AI 模型（DistilBERT + Token Classification）

脚本：`train_keyword_model.py`  
用途：在 Hugging Face `midas/inspec` 的 **关键词抽取数据集** 上训练一个小模型，用来对任意英文句子 / 段落打 BIO 标签并抽取关键词短语。

运行一次训练：

```bash
python train_keyword_model.py
```

你会在终端看到类似日志：

- 加载 tokenizer 和 Inspec 数据集  
- 训练若干 epoch（默认 5）  
- 在验证集上选最好的模型  
- 在 `./keyword-bert-inspec/` 目录下保存模型和 tokenizer

训练完成后，会生成：

```text
keyword-bert-inspec/
  ├── config.json
  ├── model.safetensors
  ├── vocab.txt
  ├── tokenizer.json
  └── tokenizer_config.json
```

> **在服务器上训练 → 本地使用：**
> - 在服务器上跑完 `train_keyword_model.py`
> - 把整个 `keyword-bert-inspec/` 文件夹拷贝到你本地项目里  
> - 本地环境中只需要安装 `torch + transformers` 等依赖，就可以直接加载并用来抽关键词，无需重新训练。

---

### 4.2 AI 选词核心逻辑（简要说明）

相关脚本：

- `keyword_extractor.py`  
  - 封装了 `extract_keywords_from_sentence()` / `extract_keywords_from_text()`  
  - 内部调用 DistilBERT 关键词模型，对句子 / 段落做 BIO 标注，合并成关键词短语。

- `candidate_builder.py`  
  - 从整篇文章中统计词频、过滤过短/过长、过滤全大写缩写等  
  - 可以按频率排序，选出候选词列表。

- `ai_select_wordlist.py`  
  - 把 **模型打分** + **频率信息** + **一些规则（过滤专有名词、奇怪符号等）** 结合起来  
  - 得到一份适合“生词本”的候选词表  
  - 支持：
    - `ai_top_n` 为整数：取前 N 个关键词  
    - `ai_top_n` 为 0–1 小数：按文章中不同词的数量的百分比取词（例如 0.1 = 10%）

---

## 5. 一键流水线：从 PDF 直接到 词汇 JSON/CSV

脚本：`pipeline_from_pdf.py`  
用途：**从一个 PDF 文件直接跑完整套流程**：

- 提取 PDF 文本 → `.txt`
- AI 或 手动模式选词 → `.select.txt` / `.ai.select.txt`
- 调用 `build_vocab_combined.py` 查询词典 → `.csv`
- 再把 CSV 转成 JSON → `.json`
- 所有结果都按 **文章名** 放进一个独立文件夹，方便前端 / App 使用

### 5.1 输出目录结构示例

以 `data/Xuanzang-page 1-5.pdf` 为例，脚本会生成：

```text
reading_Xuanzang-page_1-5/
  ├── Xuanzang-page_1-5.txt           # 从 PDF 提取的全文 txt
  ├── Xuanzang-page_1-5.ai.select.txt # AI 挑选出的生词列表（原始候选）
  ├── Xuanzang-page_1-5.ai.words.txt  # 出现在文章中的目标词列表
  ├── Xuanzang-page_1-5.ai.csv        # 含释义 & 例句的词汇表（给 Excel 用）
  └── Xuanzang-page_1-5.ai.json       # 同样内容的 JSON（给网页 / App 直接用）
```

> 以后前端只要读取这个 `reading_XXX` 文件夹下面的 `.json`，就能给学生展示“本篇文章的生词本”。

---

## 6. 使用方式详解

### 6.1 模式一：AI 自动选词（推荐）

适合：  
- 没有现成单词表  
- 想让 AI 根据文章内容自动猜哪些词比较难

命令示例：

```bash
python pipeline_from_pdf.py \
  --mode ai \
  --pdf "data/Xuanzang-page 1-5.pdf" \
  --ai_top_n 0.1
```

参数说明：

- `--mode ai`：使用 AI 选词模式  
- `--pdf`：PDF 文件路径  
- `--ai_top_n`：
  - 如果是 **整数**（例如 `30`）：取前 30 个关键词
  - 如果是 **小数**（0–1，如 `0.1`）：按“文章中不同单词的数量 × 0.1”来取，等于“抽取 10% 的生词”

例如：

```bash
# 取前 40 个关键词
python pipeline_from_pdf.py --mode ai --pdf "data/xxx.pdf" --ai_top_n 40

# 按 15% 比例取词
python pipeline_from_pdf.py --mode ai --pdf "data/xxx.pdf" --ai_top_n 0.15
```

运行完成后，你会看到终端类似输出：

- [Step 1] 从 PDF 提取文本  
- [Step 2] 用 AI 生成生词表（显示抽取了多少个词）  
- [Step 3] 构建词汇表 CSV（逐个单词显示是否查到释义 / 例句）  
- [Step 4] 转成 JSON  
- 最后提示：所有文件都在 `reading_文章名` 文件夹中

---

### 6.2 模式二：使用自己准备的单词表

适合：

- 教师 / 学生已经有一份“重点单词表（比如考试范围）”
- 只想在文章里筛选“这篇文章真正出现过的单词”

命令示例：

```bash
python pipeline_from_pdf.py \
  --mode list \
  --pdf "data/YourArticle.pdf" \
  --select "data/your_word_list.txt"
```

- `--mode list`：表示“列表模式”（不用 AI 选词）  
- `--select`：你自己的单词列表（每行一个词）

输出类似：

```text
reading_YourArticle/
  ├── YourArticle.txt            # PDF 提取出来的文本
  ├── YourArticle.list.words.txt # 在文章中真正出现过的目标词列表
  ├── YourArticle.list.csv       # 词汇表 CSV
  └── YourArticle.list.json      # 词汇表 JSON
```

---

## 7. CSV & JSON 字段说明

无论是 AI 模式还是手动模式，最终的 CSV/JSON 都有三个核心字段：

- `word` / `﻿word`：目标词
- `meaning`：英文释义（来自 Merriam-Webster，如可用）
- `example`：英文例句（来自 Cambridge，如可用；否则会退回到文章中的原句）

示例 JSON（缩写）：

```json
[
  {
    "word": "auspicious",
    "meaning": "showing or suggesting that future success is likely : propitious",
    "example": "They won their first match of the season 5–1 which was an auspicious start/beginning."
  },
  {
    "word": "disposition",
    "meaning": "the usual attitude or mood of a person or animal",
    "example": "She is of a nervous/cheerful/sunny disposition."
  }
]
```

前端 / App 可以直接用这个 JSON 来展示生词卡片。

---

## 8. 在服务器训练，在本地使用模型的流程总结

1. **在实验室服务器上：**

   ```bash
   # 激活环境
   conda activate article-helper

   # 进入项目目录
   cd article-reading-helper

   # 训练一次模型
   python train_keyword_model.py
   ```

   训练完成后会生成 `keyword-bert-inspec/` 目录。

2. **把模型复制到本地电脑：**

   - 用 `scp`、SFTP、WinSCP 等工具，把整个 `keyword-bert-inspec/` 文件夹下载到你本地项目目录中。

3. **在本地电脑上：**

   ```bash
   conda create -n article-helper python=3.10
   conda activate article-helper
   pip install torch transformers datasets requests beautifulsoup4 pandas seqeval
   ```

   然后就可以直接在本地跑：

   ```bash
   python pipeline_from_pdf.py \
     --mode ai \
     --pdf "data/YourArticle.pdf" \
     --ai_top_n 0.1
   ```

   本地并不会再训练，只是 **加载现成的模型** 来选词。

---

## 9. 未来扩展 (Future Work Ideas)

- 根据学生水平（如 CEFR / TOEFL 词频）动态调节“生词难度阈值”
- 在前端让学生点“我会 / 我不会”，回写一个“个人词库”，下次自动跳过会的词
- 接上更大的多模态模型，支持“图文混合”的阅读材料
- 增加中文释义 / 双语例句（接入其他词典 API）

---

如果你只想快速开始：

1. 把 PDF 放到 `data/` 里  
2. （可选）在服务器上先跑 `python train_keyword_model.py` 训练一次  
3. 之后直接：

```bash
python pipeline_from_pdf.py --mode ai --pdf "data/YourArticle.pdf" --ai_top_n 0.1
```

拿到 `reading_YourArticle/YourArticle.ai.json`，就可以连到网页 / App 了 🎯
