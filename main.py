import json

import data_processor as dp
import ner.ner_runner as ner
import ocr as ocr
import img_processor as imgp

# Class to store configuration for this pipeline run
class Parameters:
    def __init__(self):
        self.pdfConverter = dp.PdfConverter.pymupdf
        self.ocrMethod = ocr.OcrMethod.pytesseract
        self.nerMethod = ner.NerMethod.key_words_extractor

params = Parameters()

# Load all files (PDFs/images) from the data folder and convert PDFs to images
data_images = dp.get_images(False, params.pdfConverter)

# Loop through all invoices/images
for invoice in data_images:
    print('Process invoice ' + invoice)
    # Take the first page/image for each file
    image = data_images[invoice][0]

    # Step 1: Run OCR on the image to extract text
    print('Run OCR')
    text, ocr_result = ocr.run_ocr(image, params.ocrMethod)

    # Step 2: Run NER to extract structured entities
    print('Run NER')
    response = ner.run_ner(image, text, ocr_result, dp.path_model, params.nerMethod)
    result_dict = json.loads(response)

    # Step 3: Save structured NER output to JSON file
    dp.write_result_json(invoice, result_dict)

    # Step 4: Prepare data for visualization
    # Extract all non-empty values from NER result
    words = [result_dict[field] for field in result_dict if result_dict[field] is not None]
    words = [str(word) for word in words if len(word) > 0]
    # Create a mapping from word value to field name for labels
    titles_dict = {v: k for k, v in result_dict.items() if v in words}

    # Step 5: Draw bounding boxes and labels on the original image
    boxed_img = imgp.draw_boxes(image, ocr_result, words, titles_dict)

    # Step 6: Save the annotated image to the output folder
    dp.write_image(invoice, boxed_img)
