from transformers import DonutProcessor, AutoModelForImageTextToText
import os
import re
import json
import timeit

class DonutRunner:
    """
    Wrapper class to load, run, and process the Donut model
    """
    def __init__(self, path_model):
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.model_id = 'to-be/donut-base-finetuned-invoices'
        self.current_path = os.path.abspath(os.getcwd())
        self.save_path = os.path.join(path_model, 'donut-base-finetuned-invoices')
        self.model_loaded = False


    def load_or_download_model(self):
        # Check if model folder exists and contains config file
        if os.path.exists(os.path.join(self.save_path, "config.json")):
            print("Loading donut model from local directory.")
            model = AutoModelForImageTextToText.from_pretrained(self.save_path)
            processor = DonutProcessor.from_pretrained(self.save_path)
        else:
            print("Model not found locally. Downloading.")
            model = AutoModelForImageTextToText.from_pretrained(self.model_id)
            processor = DonutProcessor.from_pretrained(self.model_id)

            # Save into your directory
            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            processor.save_pretrained(self.save_path)

            print("Model downloaded and saved to: ", self.save_path)

        return model, processor

    def prepare_model(self):
        self.model, self.processor = self.load_or_download_model()


    def run_donut(self, image):
        # Ensure model is loaded only once
        if not self.model_loaded:
            self.prepare_model()
            self.model_loaded = True

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Generate predicted output
        outputs = self.model.generate(pixel_values)
        predicted_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract key-value pairs
        matches = re.findall(r"<s_(\w+)>\s*(.*?)\s*</s_\1>", predicted_text)

        dict_res = {}
        for key, value in matches:
            dict_res[key] = value

        # Map Donut keys to human-readable field names
        result = {}
        if 'InvoiceNumber' in dict_res: result["Invoice Number"] = dict_res['InvoiceNumber']
        if 'TaxAmount1' in dict_res: result["Tax"] = dict_res['TaxAmount1']
        if 'GrossAmount' in dict_res: result["Total"] = dict_res['GrossAmount']
        if 'DocumentDate' in dict_res: result["Invoice date"] = dict_res['DocumentDate']

        print('Found entities: ')
        print(result)

        return json.dumps(result)

