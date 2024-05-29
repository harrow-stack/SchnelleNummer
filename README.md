# SchnelleNummer
Testing some ocr in python (currently using easyocr only "0123456789")

install requirements:

    pip install -r requirements.txt


This script loads a video and takes a screenshot every *x* seconds.
Choose one or multiple areas to be read.

## Sample Command

    python main.py -video test.mp4 -interval 5 -out result.csv

Reads *test.mp4* and OCR every *5* seconds -> write results to *result.csv*

    python main.py -h


list all commands

--- 
### Change OCR
You can swap easyocr with every python ocr.
Simply edit *get_numbers_ocr()*. It gets an img as an array and should return only the text it recognized.

---
