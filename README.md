# article-reading-helper

An online article-reading helper that turns a PDF into:

- clean text
- an AI-selected vocabulary list
- meanings (Merriam-Webster) + example sentences (Cambridge)
- JSON that can be consumed by a website or app

The idea:  
A student picks an article → the system extracts the text → an AI model guesses which words are hard → dictionary APIs/web-scrapers fetch meanings & examples → the result is saved as CSV + JSON for the reading app.

---

## 1. Main Features

- **PDF → TXT**
  - Extract raw text from a PDF file.
- **Vocabulary builder (classic mode)**
  - Given:
    - an article (`.txt`)
    - a user word list (`.txt`)
  - Build:
    - a list of matched words in the article
    - a CSV with meaning + example
    - an optional JSON file for front-end use.
- **AI word selection**
  - Train a BERT-style keyword model on the **midas/inspec (extraction)** dataset.
  - Use the model to automatically pick “likely difficult” words from an article.
- **One-shot pipeline from PDF**
  - `PDF → TXT → AI wordlist → CSV → JSON`
  - Everything saved into a folder named `reading_{article_name}`.

---

## 2. Repository Layout

Typical layout (your repo may look like this):

```text
article-reading-helper/
├── ai_select_wordlist.py        # AI-based word selection (from article text)
├── build_vocab_combined.py      # Build vocab CSV from article + word list
├── candidate_builder.py         # Build candidate words (frequency / filters)
├── config.py                    # Central configuration (paths, API keys, etc.)
├── csv_to_json.py               # Convert vocab CSV → JSON for front-end
├── data/                        # Put your PDFs and sample data here
├── extract_pdf_text.py          # PDF → TXT helper
├── keyword_extractor.py         # Low-level keyword extraction functions
├── keyword-bert-inspec/         # Trained keyword model folder (output)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── llm_refiner.py               # (Optional) LLM-based fallback refiner
├── pipeline_from_pdf.py         # Full pipeline script (PDF → JSON)
├── text_utils.py                # Text cleaning / tokenization helpers
└── train_keyword_model.py       # Train the keyword model (once)
```

---

## 3. Installation

### 3.1. Python environment

Recommended:

- Python 3.9+  
- Conda or virtualenv

Example with `conda`:

```bash
conda create -n article-helper python=3.10
conda activate article-helper
```

### 3.2. Dependencies

Install the main libraries (you can adjust as needed):

```bash
pip install torch transformers datasets seqeval requests beautifulsoup4 lxml PyPDF2
```

If you do not care about training metrics (F1), you can skip `seqeval`.

---

## 4. Training the Keyword Model (once)

You usually run this once on a powerful server, then copy the trained model folder to your laptop.

Script: `train_keyword_model.py`

High-level:

- Uses `distilbert-base-cased` (or similar) from Hugging Face.
- Trains a token classification model (BIO tags) on `midas/inspec` (`"extraction"` config).
- Saves the best model to `./keyword-bert-inspec`.

### 4.1. Run training

From the project root:

```bash
python train_keyword_model.py
```

What it does:

1. Downloads the Inspec extraction dataset.
2. Downloads the base transformer model.
3. Trains for several epochs (defaults are in the script: `EPOCHS`, `BATCH_SIZE`, `LR`, etc.).
4. Saves the best validation loss model into:

```text
./keyword-bert-inspec/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    vocab.txt
```

You should see logs like:

- `===== Epoch 1/5 =====`
- `Train loss: ...`
- `Val loss: ...`
- `Test metrics: {...}`
- `Training finished, best model saved to ./keyword-bert-inspec`

### 4.2. Using the trained model on another machine

To run the AI pipeline on your own laptop:

1. Copy the entire `keyword-bert-inspec/` folder from the server to your local repo root (same level as `pipeline_from_pdf.py`).
2. On your laptop, install the Python dependencies.
3. Run the pipeline (see section 5).  
   The code will automatically load `./keyword-bert-inspec` if it exists.

As long as the relative path stays the same, you can freely move between machines.

---

## 5. One-Shot Pipeline: PDF → TXT + AI Wordlist + CSV + JSON

Script: `pipeline_from_pdf.py`

General form:

```bash
python pipeline_from_pdf.py   --mode {ai|list}   --pdf path/to/your_file.pdf   [--select wordlist.txt]   [--ai_top_n VALUE]
```

All outputs are placed in a folder named:

```text
reading_{PDF_STEM}/
```

where `PDF_STEM` is the PDF filename without extension (and with some simple cleaning of spaces and special characters).

Example:  
`data/Xuanzang-page 1-5.pdf` → `reading_Xuanzang-page_1-5/`

### 5.1. Output files (AI mode)

In AI mode, you typically get:

```text
reading_Xuanzang-page_1-5/
├── Xuanzang-page_1-5.txt              # extracted article text
├── Xuanzang-page_1-5.ai.select.txt    # AI-selected raw word list (one per line)
├── Xuanzang-page_1-5.ai.words.txt     # words actually found in the article
├── Xuanzang-page_1-5.ai.csv           # word, meaning, example (for spreadsheet)
└── Xuanzang-page_1-5.ai.json          # same info, JSON (for web/app)
```

The pipeline steps:

1. **Step 1 – Extract text from PDF**

   Uses `extract_pdf_text.py` internally:

   - Reads every page.
   - Writes a cleaned `.txt` file into the reading folder.

2. **Step 2 – AI select words**

   Uses:

   - `keyword_extractor.py` and the trained model (`keyword-bert-inspec`)
   - `candidate_builder.py` for candidate filtering (stopwords, frequency, casing)
   - Internal filtering rules to drop:
     - words with apostrophes (for example `"xuanzang's"`)
     - tokens that contain two or more hyphens (for example `character--whether`)
   - It may also skip some very rare or broken tokens with mostly non-alphabetic characters.

3. **Step 3 – Build vocab CSV**

   Uses `build_vocab_combined.py` logic to:

   - Check if each candidate word appears in the article text.
   - Query Merriam-Webster for a meaning.
   - Query Cambridge Dictionary for an example sentence.
   - If Merriam-Webster fails but a meaning is found via a fallback lookup, that fallback is used.
   - If no standard dictionary definition is found, the word is still kept with a short note such as  
     `"No standard dictionary definition found; see example for context."`  
     This is useful for proper nouns or technical terms.
   - Writes a CSV file.

4. **Step 4 – Convert CSV → JSON**

   Uses `csv_to_json.py`:

   - Reads the CSV from step 3.
   - Writes a JSON file with an array of objects:
     ```json
     {
       "word": "monk",
       "meaning": "a man who is a member of a religious order and lives in a monastery",
       "example": "..."
     }
     ```
   - This JSON is what you will typically load in your web or mobile app.

### 5.2. Controlling how many words AI selects (`--ai_top_n`)

The flag `--ai_top_n` controls how many AI-ranked words are kept.

You can pass either:

- Integer K: “keep the top K words”

  ```bash
  # keep at most 30 words
  python pipeline_from_pdf.py     --mode ai     --pdf data/Xuanzang-page_1-5.pdf     --ai_top_n 30
  ```

- Float between 0 and 1: “keep this percentage of content words”

  ```bash
  # keep top 10% of content words
  python pipeline_from_pdf.py     --mode ai     --pdf data/Xuanzang-page_1-5.pdf     --ai_top_n 0.10
  ```

Internally the script:

1. Collects candidate “content words” after filtering stopwords and junk.
2. Sorts them by model score.
3. If `ai_top_n >= 1`, it keeps the first `int(ai_top_n)` words.  
   If `0 < ai_top_n < 1`, it keeps approximately that fraction of candidates (rounded).

---

## 6. Classic Mode: Use Your Own Word List

If you already have a custom word list (for example `20241226.txt`) and just want to match words in an article and build a CSV/JSON, you can use list mode.

### 6.1. `extract_pdf_text.py` (PDF → TXT)

Simple helper:

```bash
python extract_pdf_text.py --input path/to/article.pdf --output article.txt
```

This creates `article.txt`.

### 6.2. `build_vocab_combined.py` (article + word list → vocab CSV)

Purpose:  
Use a given word list to scan the article and build a vocabulary list with dictionary information.

Command:

```bash
python build_vocab_combined.py   --article article.txt   --select your_word_list.txt   --out_words words.txt   --out_csv words.csv
```

- `article` – text file of the article
- `select` – list of target words (one per line)
- `out_words` – subset of `select` that actually appears in the article
- `out_csv` – vocabulary CSV with meaning and example

### 6.3. `csv_to_json.py` (CSV → JSON)

Once you have `words.csv`:

```bash
python csv_to_json.py words.csv words.json
```

This produces `words.json` which your website or app can load.

### 6.4. One-shot from PDF with your word list

You can also use `pipeline_from_pdf.py` in list mode to avoid multiple manual steps:

```bash
python pipeline_from_pdf.py   --mode list   --pdf data/Xuanzang-page_1-5.pdf   --select my_word_list.txt
```

Output will again be under:

```text
reading_Xuanzang-page_1-5/
    Xuanzang-page_1-5.txt
    Xuanzang-page_1-5.list.words.txt
    Xuanzang-page_1-5.list.csv
    Xuanzang-page_1-5.list.json
```

---

## 7. Notes and Limitations

- Dictionary lookups rely on the current HTML from Merriam-Webster and Cambridge.
  - If they change their page structure, scraping logic may need to be updated.
- Some very rare words, transliterations, or technical religious terms may not exist in standard dictionaries.
  - In those cases, the pipeline keeps the word and uses:
    - the example sentence from the article itself, and/or
    - a short fallback message instead of a standard definition.
- `ai_top_n` is heuristic.
  - For academic or historical texts with many proper nouns, you may want to:
    - use a smaller percentage (for example 0.05 → 5%), or
    - combine AI selection with a hand-curated word list.

---

## 8. Typical Workflow Summary

1. Train the model once (on a server):

   ```bash
   python train_keyword_model.py
   # outputs ./keyword-bert-inspec/
   ```

2. Copy `keyword-bert-inspec/` to your local repo (if needed).

3. For each new article PDF:

   ```bash
   python pipeline_from_pdf.py      --mode ai      --pdf data/SomeArticle.pdf      --ai_top_n 0.10
   ```

4. Load `reading_SomeArticle/SomeArticle.ai.json` in your web or mobile app and display:

   - word
   - meaning
   - example

Students can click a word in the article and see the definition and example directly on the side.
