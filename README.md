# article-reading-helper
An online article reading helper that extracts text from PDFs and builds a study-ready vocabulary list from an article using a given word list.

---

## Features
- Convert **PDF → TXT** for easy processing
- Build vocabulary CSV from an article + selected word list
- **Meaning** from *Merriam-Webster*, **Example sentence** from *Cambridge Dictionary*
- Windows-friendly (notes for PowerShell included)

---

## Directory Layout


**help/extract_pdf_text.py:**

purpose: make the pdf file into txt file

python extract_pdf_text.py --input name.pdf --output name.txt

---
**software/build_vocab_combined.py:**

purpose: using the given word list file to check the article. The meaning is given by the Merriam-Webster. The example sentence is given by Cambridge Dictionary

python build_vocab_combined.py --article name.txt --select the_word_list.txt --out_words words.txt --out_csv words.csv
