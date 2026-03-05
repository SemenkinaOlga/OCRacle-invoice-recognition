from enum import Enum

from ner.local import donut, mychen76_donut as mcd, local_llm as lllm, keywords_extraction as kwe
import data_processor as dp
import test_ner as tn


# Enum to select the NER method
class NerMethod(Enum):
    mistral_llm = 1
    donut = 2
    mychen_donut = 3
    key_words_extractor = 4
    test = 999

# Initialize DonutRunner with the model path from data_processor
donut_runner = donut.DonutRunner(dp.path_model)
mc_donut_runner = mcd.MychenDonutTestRunner(dp.path_model)
test_runner = tn.TestRunner(dp.path_model)
keyWordsExtractor = kwe.KeyWordsExtractor()

# Wrapper function to run NER based on selected method
def run_ner(image, text: str, ocr_result, path, ner_method: NerMethod = NerMethod.mistral_llm):
    if ner_method == NerMethod.mistral_llm:
        llm_params = lllm.LlmParameters(path)
        return lllm.ask_local_llm(llm_params, text)
    elif ner_method == NerMethod.donut:
        return donut_runner.run_donut(image)
    elif ner_method == NerMethod.mychen_donut:
        return mc_donut_runner.run_donut(image, text)
    elif ner_method == NerMethod.key_words_extractor:
        return keyWordsExtractor.run(ocr_result)
    elif ner_method == NerMethod.test:
        return test_runner.run_test_model(image, text, ocr_result)
    else:
        return []