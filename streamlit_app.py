import streamlit as st
from PIL import Image
from io import BytesIO
import json
import os
import traceback
import time
import invoice_recognition as inv_rec
import data_processor as dp
import ner.ner_runner as ner
import ocr as ocr
import img_processor as imgp


params = inv_rec.Parameters()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2000  # pixels

st.set_page_config(layout="wide", page_title="OCRacle-invoice-recognition")

st.write("## Get data from your invoice by OCRacle")
st.write(
    "Try uploading an invoice to watch what OCR and NER can find."
)

st.write(
    "This app is open source and code available on [GitHub](https://github.com/SemenkinaOlga/OCRacle-invoice-recognition). Created by [Olga Semenkina](https://www.linkedin.com/in/olga-semenkina/)."
)

def convert_image(img):
    buf = BytesIO()
    img = Image.fromarray(img)
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def get_file_bytes(upload):
    if isinstance(upload, str):
        # Default path
        if not os.path.exists(upload):
            st.error(f"Default image not found at path: {upload}")
            return None, None
        file_ext = os.path.splitext(upload)[1].lower().lstrip(".")
        with open(upload, "rb") as f:
            return f.read(), file_ext
    else:
        file_ext = upload.name.split(".")[-1].lower()  # e.g. "png", "jpg", "pdf"
        return upload.getvalue(), file_ext

def fix_image(upload):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading image...")
        progress_bar.progress(10)

        file_bytes, file_ext = get_file_bytes(upload)

        status_text.text("Processing image...")
        progress_bar.progress(20)

        image = dp.get_image(file_bytes, file_ext)

        status_text.text("Run OCR...")
        progress_bar.progress(30)

        text, ocr_result = ocr.run_ocr(image, params.ocrMethod)

        status_text.text("Run NER...")
        progress_bar.progress(60)

        response = ner.run_ner(image, text, ocr_result, dp.path_model, params.nerMethod)
        result_dict = json.loads(response)

        with st.expander("📄 Parsed Invoice Data", expanded=True):
            st.json(result_dict)

        status_text.text("Prepare results...")
        progress_bar.progress(90)

        # Prepare data for visualization
        # Extract all non-empty values from NER result
        words = [result_dict[field] for field in result_dict if result_dict[field] is not None]
        words = [str(word) for word in words if len(word) > 0]
        # Create a mapping from word value to field name for labels
        titles_dict = {v: k for k, v in result_dict.items() if v in words}

        # Draw bounding boxes and labels on the original image
        boxed_img = imgp.draw_boxes(image, ocr_result, words, titles_dict)

        # Display images
        col1.write("### Original Image")
        col1.image(image)

        col2.write("### What was found")
        col2.image(boxed_img)

        # Prepare download button
        st.sidebar.markdown("\n")
        st.sidebar.download_button(
            "Download fixed image",
            convert_image(boxed_img),
            "fixed.png",
            "image/png"
        )

        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process image")
        # Log the full error for debugging
        print(f"Error in fix_image: {traceback.format_exc()}")

st.sidebar.write("## Settings")

st.sidebar.write("## Upload and download")

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an invoice", type=["png", "jpg", "jpeg", "pdf"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload a file smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        fix_image(upload=my_upload)
else:
    # Try default images in order of preference
    default_image = "./data/1_invoice_example.png"
    if os.path.exists(default_image):
        fix_image(default_image)
    else:
        st.info("Please upload an image to get started!")