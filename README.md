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

### 🌐 Streamlit Web App

In addition to the command-line interface, OCRacle now includes an interactive **Streamlit web app** for uploading and processing invoices through a browser UI.

#### Run locally:
```
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

#### Try it online:

The app is also deployed on **Streamlit Community Cloud** — no installation needed:

👉 [Open OCRacle on Streamlit Cloud](https://ocracle.streamlit.app/)

### 🧰 Features
* 📄 **Flexible PDF & Image Input** – Supports PDF (via pdf2image or PyMuPDF) and image files using Pillow.


* 🔎 **OCR Processing** – Extracts text from invoices with Tesseract (pytesseract).


* 🧠 **NER Extraction** (2 Options)
    * Local quantized Mistral LLM via LangChain

    * Transformer model Donut (donut-base-finetuned-invoices) from Hugging Face (auto-downloaded on first run)

    * Transformer model Donut fine-tuned for invoices (mychen76/invoice-and-receipts_donut_v1) from Hugging Face (auto-downloaded on first run)
  
    * LayoutLM for Invoices from Hugging Face impira/layoutlm-invoices (auto-downloaded on first run)

    * Complex custom RegEx patterns to parse fields 'Invoice number', 'Invoice date', 'Tax', 'Shipping' and 'Total' from ocr results

* 📦 **Structured Output** – Saves extracted invoice data as JSON per document.


* 🖼️ **Visual Annotations** – Generates images with bounding boxes (OpenCV), highlighting detected fields.

* 🌐 **Interactive Web UI** – Browser-based interface via Streamlit for uploading invoices 
and visualizing extracted fields.

### ⚠️ Known Issues
* 🤖 **Mistral (LLM) Instability**

    * May ignore strict JSON output format.
  
    * Local LLM works slowly, even when quality is already sacrificed through quantization.

    * Sometimes requires multiple runs (e.g., 3 attempts) to obtain valid JSON.

    * Unpredictable named entities extraction.

    * Risk of hallucinations in extracted fields.
  

* 🍩 **Donut Performance**

    * Low accuracy despite invoice fine-tuning.

    * Struggles with diverse invoice layouts and noisy images.
  
  
* 🛠️ **Code Quality**

    * Currently, processes only the first page/image of a PDF, and doesn't fully support multipage invoices.
  
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

    * Test alternative LLMs and transformers.

    * Improve prompt engineering (strict schema enforcement, double prompting, etc.).

    * Optimize output validation with automatic retry and correction prompts.

    * Fine-tune a custom LLM on a domain-specific dataset.


* 🤝 **Ensemble Strategy**

    * Combine multiple OCR/NER methods.

    * Use voting or confidence aggregation to determine final entity values.

  

### 🔬 Other tried methods
* **spaCy NER** – Classic NLP library, but gave poor results on example invoices. Requires downgrade to Python 3.12 due to dependency issues.

* **Ollama LLMs** – Solid framework for local LLMs, but requires separate installation alongside Tesseract.

* **DocTR OCR** – Slightly lower accuracy than Tesseract (missed dates and invoice numbers), but easier to use.