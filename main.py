import invoice_recognition as inv_rec
import data_processor as dp
import ner.ner_runner as ner
import ocr as ocr

params = inv_rec.Parameters(dp.PdfConverter.pymupdf, ocr.OcrMethod.pytesseract, ner.NerMethod.key_words_extractor)

# Load all files (PDFs/images) from the data folder and convert PDFs to images
data_images = dp.get_images(False, params.pdfConverter)

inv_rec.run_invoice_recognition(data_images, params)
