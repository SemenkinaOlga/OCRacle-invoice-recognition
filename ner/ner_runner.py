from enum import Enum

from ner.local import donut, mychen76_donut as mcd, local_llm as lllm, keywords_extraction as kwe
import data_processor as dp
from ner.local import impira_layoutlm_invoices as impra


# Enum to select the NER method
class NerMethod(Enum):
    mistral_llm = 1
    donut = 2
    mychen_donut = 3
    key_words_extractor = 4
    impra_layout = 5

# Initialize runners with the model path from data_processor
donut_runner = donut.DonutRunner(dp.path_model)
mc_donut_runner = mcd.MychenDonutTestRunner(dp.path_model)
keyWordsExtractor = kwe.KeyWordsExtractor()
impra_runner = impra.PipelineRunner(dp.path_model)

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
    elif ner_method == NerMethod.impra_layout:
        return impra_runner.run_model(image)
    else:
        return []