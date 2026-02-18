# OCRacle-invoice-recognition

### 🧩 Prerequisites

- **Python 3.14.2+**
- **Tesseract 5.5.2+** (See how to [install](https://tesseract-ocr.github.io/tessdoc/Installation.html))

### 🚀 Quickstart

1. Clone the repository

``` 
git clone https://github.com/SemenkinaOlga/OCRacle-invoice-recognition.git
```


2. Install the requirements:

```
pip install -r requirements.txt
```


3. Create folders ```data```, ```model``` and ```output``` in the repository root.


4. Copy invoice files to the ```data``` folder.
_Supported formats: .pdf, .jpg, .png_


5. Download pretrained quantized Mistral model [mistral-7b-instruct-v0.1.Q5_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf) from [huggingface](https://huggingface.co/),
and put the model file in the ```model``` folder.


6. Run the main script

```
python main.py
```

### 🧰 Features
* 📄 **Flexible PDF & Image Input** – Supports PDF (via pdf2image or PyMuPDF) and image files using Pillow.


* 🔎 **OCR Processing** – Extracts text from invoices with Tesseract (pytesseract).


* 🧠 **NER Extraction** (2 Options)
    * Local quantized Mistral LLM via LangChain

    * Transformer model Donut (donut-base-finetuned-invoices) from Hugging Face (auto-downloaded on first run)


* 📦 **Structured Output** – Saves extracted invoice data as JSON per document.


* 🖼️ **Visual Annotations** – Generates images with bounding boxes (OpenCV), highlighting detected fields.



### ⚠️ Known Issues
* 🤖 **Mistral (LLM) Instability**

    * May ignore strict JSON output format.
  
    * Local LLM works slowly, even when quality is already sacrificed through quantization.

    * Sometimes requires multiple runs (e.g., 3 attempts) to obtain valid JSON.

    * Unpredictable named entities extraction.

    * Risk of hallucinations in extracted fields.
  

* 🍩 **Donut Performance** (donut-base-finetuned-invoices) 

    * Low accuracy despite invoice fine-tuning.

    * Struggles with diverse invoice layouts and noisy images.
  
  
* 🛠️ **Code Quality**

    * Currently processes only the first page/image of a PDF, and doesn't fully support multipage invoices.
  
    * Code needs refactoring for clarity and maintainability.
  
    * Lacks proper code documentation and inline explanations.
  
    * Code is currently using an outdated version of the LangChain library and needs to be upgraded to the latest version.



### 📈 Future Improvements

* 🖼️ **Image Preprocessing Enhancements**

    * Use OpenCV for denoising, thresholding, contrast adjustment, and other.

    * Experiment with other frameworks to improve OCR quality.
  

* 👁️ **Alternative OCR Engines**

    * Try other OCR frameworks for better accuracy (EasyOCR, LayoutParser, Camelot, tabula, etc.).


* 📦 **Cleaner Visual Output**

    * Improve bounding box logic to prevent duplicate/overlapping highlights.

    * Add confidence-based filtering for drawn entities.


* 🧠 **NER Improvements**

    * Test alternative LLMs better suited for structured extraction (LayoutLM, LayoutXLM, LayoutLMv3).

    * Improve prompt engineering (strict schema enforcement, double prompting, etc.).

    * Optimize output validation with automatic retry and correction prompts.

    * Fine-tune a custom LLM on a domain-specific dataset.

    * Replace Donut with more robust document-understanding models fine-tuned for invoices.


* 🤝 **Ensemble Strategy**

    * Combine multiple OCR/NER methods.

    * Use voting or confidence aggregation to determine final entity values.

  

### 🔬 Other tried methods
* **spaCy NER** – Classic NLP library, but gave poor results on example invoices. Requires downgrade to Python 3.12 due to dependency issues.

* **Python re & Custom Rules** – Works only for limited, fixed invoice templates. Fails on varied formats.

* **Ollama LLMs** – Solid framework for local LLMs, but requires separate installation alongside Tesseract.

* **DocTR OCR** – Slightly lower accuracy than Tesseract (missed dates and invoice numbers), but easier to use.