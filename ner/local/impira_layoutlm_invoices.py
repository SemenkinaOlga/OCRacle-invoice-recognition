from transformers import pipeline
import os
import json

class PipelineRunner:
    def __init__(self, path_model):
        self.pipeline = None
        self.model_id = 'impira/layoutlm-invoices'
        self.save_path = os.path.join(path_model, "impira-layoutlm-invoices")
        self.model_cached = any(self.model_id.replace("/", "--") in d for d in os.listdir(self.save_path)) if os.path.exists(
            self.save_path) else False
        self.model_loaded = False
        self.CONFIDENCE_THRESHOLD = 0.5


    def prepare_model(self):
        if not self.model_loaded:
            self.pipeline = pipeline(
                "document-question-answering",
                model=self.model_id,
                device="cpu",
                cache_dir=self.save_path,  # <-- custom download location
                local_files_only=self.model_cached,  # force local if already downloaded
            )
            self.model_loaded = True


    def run_model(self, image):
        self.prepare_model()

        questions = {
            "Invoice Number": "What is the invoice number?",
            "Invoice Date": "What is the invoice date?",
            "Total": "What is the total amount?",
            "Tax": "What is the tax amount?",
            "Shipping": "What is the shipping cost?",
        }

        output = {}
        for field, question in questions.items():
            answer = self.pipeline(image, question=question)
            score = answer[0]["score"]
            value = answer[0]["answer"]

            if score >= self.CONFIDENCE_THRESHOLD:
                output[field] = value

        return json.dumps(output)

