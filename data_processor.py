from PIL import Image
import os
import re
from pdf2image import convert_from_path
from enum import Enum
import pymupdf
import json
import matplotlib.pyplot as plt

# Define folder names
data_folder = 'data'
tmp_data_folder = 'tmp_data'
output_data_folder = 'output'
model_data_folder = 'model'

# Get absolute paths for these folders relative to the current working directory
current_path = os.path.abspath(os.getcwd())
path_data = os.path.join(current_path, data_folder)
path_output = os.path.join(current_path, output_data_folder)
path_model = os.path.join(current_path, model_data_folder)

# Set up a transformation matrix for pymupdf rendering (scale 4x)
mat = pymupdf.Matrix(4, 4)

# Enum to select which PDF conversion library to use
class PdfConverter(Enum):
    pdf2image = 1
    pymupdf = 2

# Function to write a JSON file in the output folder
def write_result_json(name, result_json):
    file_name = os.path.join(path_output, name + '.json')
    with open(file_name, 'w') as f:
        json.dump(result_json, f)
    print("Created file " + file_name)

# Function to extract the first valid JSON object/array substring from a string with LLM response
def extract_json_substrings(s):
    results = []
    # Stack to keep track of opening braces/brackets
    stack = []
    # Start index of a potential JSON substring
    start_idx = None

    # Iterate through the string character by character
    for i, char in enumerate(s):
        if char in '{[':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char in '}]':
            if not stack:
                continue
            open_brace = stack.pop()
            if ((open_brace == '{' and char != '}') or
                    (open_brace == '[' and char != ']')):
                stack.append(open_brace)  # mismatched, put back
                continue
            # If stack is empty after popping, we have a complete JSON substring
            if not stack:
                candidate = s[start_idx:i + 1]
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    continue

    # Return the first JSON substring as a string, or empty JSON if none found
    if len(results) > 0:
        if len(results[0]) > 0:
            res = json.dumps(results[0])
            print(type(res))
            return res
    res = json.dumps({})
    print(type(res))
    return res

# Function to save an image to the output folder
def write_image(name, image):
    file_name = os.path.join(path_output, name + '.jpg')
    plt.imsave(file_name, image)
    print("Created file " + file_name)

# Function to get all file names in the data folder
def get_all_file_names() -> dict:
    files = {}
    for filename in os.listdir(path_data):
        path, name = os.path.split(filename)
        files[name] = os.path.join(path_data, filename)
    return files

# PDF conversion using pdf2image
def convert_pdf_to_image_pdf2image(path):
    return convert_from_path(path)

# PDF conversion using pymupdf
def convert_pdf_to_image_pymupdf(path):
    images = []
    doc = pymupdf.open(path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pixmap = page.get_pixmap(dpi=300, matrix=mat)
        images.append(pixmap.pil_image())
    return images


# Wrapper function to run PDF conversion based on selected method
def convert_pdf_to_image(path, pdf_converter:PdfConverter=PdfConverter.pdf2image):
    if pdf_converter == PdfConverter.pdf2image:
        return convert_pdf_to_image_pdf2image(path)
    elif pdf_converter == PdfConverter.pymupdf:
        return convert_pdf_to_image_pymupdf(path)
    else: return []

# Function to save a temporary image file if needed
def save_image(save_tmp_files, image, name, i):
    if save_tmp_files:
        new_path = os.path.join(tmp_data_folder, name + '_page_' + str(i) + '.jpg')
        image.save(new_path)

# Main function to get images from all files in the data folder
def get_images(save_tmp_files:bool=False, pdf_converter:PdfConverter=PdfConverter.pdf2image):
    file_paths = get_all_file_names()
    image_files = {}

    for file_name in file_paths:
        path = file_paths[file_name]
        name = file_name.split('.')[0]
        file_extension = os.path.splitext(file_name)[-1].lower()
        print("Extract file: " + file_name)

        if file_extension in {'.pdf'}:
            images = convert_pdf_to_image(path, pdf_converter)
            image_files[name] = images
            for i in range(len(images)):
                save_image(save_tmp_files, images[i], name, i)
        elif file_extension in {'.jpg'}:
            img = Image.open(path)
            image_files[name] = [img]
            save_image(save_tmp_files, img, name, 0)
        elif file_extension in {'.png'}:
            img = Image.open(path)
            # Convert PNG to JPG-like RGB format
            img = img.convert('RGB')
            image_files[name] = [img]
            save_image(save_tmp_files, img, name, 0)
        else:
            print (file_extension, " is an unknown file format.")

    return image_files
