from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, DonutProcessor, VisionEncoderDecoderModel
import os
import re
import json

class MychenDonutTestRunner:
    """
    Wrapper class to load, run, and process mychen76 Donut model
    """
    def __init__(self, path_model):
        self.model = None
        self.processor = None
        # This model wasn't trained to find invoice number :(
        self.model_id = 'mychen76/invoice-and-receipts_donut_v1'
        self.current_path = os.path.abspath(os.getcwd())
        self.save_path = os.path.join(path_model, 'mychen76-invoice-and-receipts_donut_v1')
        self.model_loaded = False


    def load_or_download_model(self):
        # Check if model folder exists and contains config file
        if os.path.exists(os.path.join(self.save_path, "config.json")):
            print("Loading mychen76 donut model from local directory.")
            model = VisionEncoderDecoderModel.from_pretrained(self.save_path)
            processor = DonutProcessor.from_pretrained(self.save_path)
        else:
            print("Model not found locally. Downloading.")
            model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
            processor = DonutProcessor.from_pretrained(self.model_id)

            # Save into your directory
            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            processor.save_pretrained(self.save_path)

            print("Model downloaded and saved to: ", self.save_path)

        return model, processor

    def prepare_model(self):
        self.model, self.processor = self.load_or_download_model()

    def run_donut(self, image, text):
        # Ensure model is loaded only once
        if not self.model_loaded:
            self.prepare_model()
            self.model_loaded = True

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_invoice>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        device = "cpu"
        self.model.to(device)
        outputs = self.model.generate(pixel_values.to(device),
                                 decoder_input_ids=decoder_input_ids.to(device),
                                 max_length=512,
                                 early_stopping=True,
                                 pad_token_id=self.processor.tokenizer.pad_token_id,
                                 eos_token_id=self.processor.tokenizer.eos_token_id,
                                 use_cache=True,
                                 num_beams=1,
                                 bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                 return_dict_in_generate=True,
                                 output_scores=True, )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        result_json = self.processor.token2json(sequence)
        if isinstance(result_json, list):
            result_json = result_json[0]

        result = {}
        if 'tax' in result_json: result["Tax"] = result_json['tax']
        if 'total' in result_json: result["Total"] = result_json['total']
        if 'date' in result_json: result["Invoice date"] = result_json['date']

        return json.dumps(result)

