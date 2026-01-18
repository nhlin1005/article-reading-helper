# -*- coding: utf-8 -*-
"""
Extracts text from a PDF file and saves it as a text file.

Usage:
  python extract_pdf_text.py --input document.pdf --output article.txt
"""

import argparse
import os
import sys

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str, output_path: str) -> bool:
    """Extract text from a PDF file and save it as a text file."""
    if not PDF_AVAILABLE:
        print("Error: PyPDF2 not installed. Cannot extract text from PDF.")
        print("Install with: pip install PyPDF2")
        return False

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return False

    try:
        print(f"Extracting text from PDF: {pdf_path}")
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Processing {num_pages} pages...")

            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:  # Only add if there's actual text
                    text += page_text + "\n"

                # Show progress
                if (page_num + 1) % 5 == 0 or (page_num + 1) == num_pages:
                    print(f"  Processed {page_num + 1}/{num_pages} pages")

        # Save extracted text
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        print(f"Text extracted to: {output_path}")
        return True

    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Extract text from a PDF file.")
    ap.add_argument("--input", required=True, help="Path to the input PDF file")
    ap.add_argument("--output", required=True, help="Path to the output text file")
    args = ap.parse_args()

    if extract_text_from_pdf(args.input, args.output):
        print("Extraction completed successfully.")
    else:
        print("Extraction failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
