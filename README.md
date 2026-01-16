# article-reading-helper
A online article reading helper


help/extract_pdf_text.py:
purpose: make the pdf file into txt file

python extract_pdf_text.py --input name.pdf --output name.txt


software/build_vocab_combined.py:
purpose: using the given word list file to check the article. The meaning is given by the Merriam-Webster
The example sentence is given by Cambridge Dictionary

python build_vocab_combined.py --article name.txt --select the_word_list.txt --out_words words.txt --out_csv words.csv
