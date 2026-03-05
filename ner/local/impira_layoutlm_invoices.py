from transformers import pipeline
from PIL import Image
import os
import re
import json

class PipelineRunner:
    def __init__(self, path_model):
        self.pipeline = None
        self.model_id = 'impira/layoutlm-invoices'
        self.save_path = os.path.join(path_model, "impira-layoutlm-invoices")
        self.model_cached = any(self.model_id.replace("/", "--") in d for d in os.listdir(self.save_path)) if os.path.exists(
            self.save_path) else False
        self.model_loaded = False


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
            "invoice_number": "What is the invoice number?",
            "date": "What is the invoice date?",
            "amount": "What is the total amount?",
            "vendor": "Who is the vendor?",
        }

        results = {}
        for field, question in questions.items():
            answer = self.pipeline(image, question=question)
            results[field] = {
                "answer": answer[0]["answer"],
                "score": round(answer[0]["score"], 4)
            }
            print(f"{field}: {results[field]['answer']} (confidence: {results[field]['score']})")

        print("\nFinal extracted fields:")
        print(results)


        return json.dumps({})

