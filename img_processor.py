import cv2
import numpy as np


# Preprocess an image for OCR
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh

# Try to find similarity in NER result and OCR word in boxes
def find_match(text, words):
    for word in words:
        if word in text:
            return True, word
    return False, ''

# Function to draw bounding boxes and labels on an image
def draw_boxes(img, ocr_result, words, titles_dict):
    """
        img: original image
        ocr_result: contains all words and boxes that OCR had found
        words: NER result, so everything that we extracted and need to show on the image
        titles_dict: how to label all words from NER result
    """
    img_copy = img.copy()
    img_copy = cv2.cvtColor(np.array(img_copy), cv2.COLOR_BGR2RGB)
    n_boxes = len(ocr_result['level'])
    padding = 3

    for i in range(n_boxes):
        # We draw a box if NER result is the exact word from OCR result, or it is substring
        text = ocr_result['text'][i]

        # If word found by OCR is in list of NER results, it is a match
        found = text in words

        # If any of words found by NER is a substring of OCR word, it is a match
        # Might be 5 instead of 5$ or 000007 instead of #000007
        match_exist, new_text = find_match(text, words)

        if found or match_exist:
            (x, y, w, h) = (ocr_result['left'][i] - padding, ocr_result['top'][i] - padding,
                            ocr_result['width'][i] + 2 * padding, ocr_result['height'][i] + 2 * padding)

            # Draw box
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (60, 200, 100), 2)

            if found: label = titles_dict[text]
            else: label = titles_dict[new_text]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw label background
            cv2.rectangle(img_copy, (x, y - th - 6),
                          (x + tw + 4, y),
                          (60, 200, 100), -1)

            # Draw label
            cv2.putText(img_copy, label,
                        (x + 2, y - 3),
                        font, font_scale,
                        (0, 0, 0), thickness,
                        cv2.LINE_AA)

    return img_copy