import pytesseract
import re
from enum import Enum

# Enum for selecting OCR method (can be extended in future)
class OcrMethod(Enum):
    pytesseract = 1

def ocr_pytesseract(image):
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text = re.sub(' +', ' ', " ".join(ocr_result['text']))
    return text, ocr_result

# Wrapper to run OCR
def run_ocr(path, ocr_method: OcrMethod = OcrMethod.pytesseract):
    if ocr_method == OcrMethod.pytesseract:
        return ocr_pytesseract(path)
    else:
        return []