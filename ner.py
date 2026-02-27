from enum import Enum

import local_llm as lllm
import donut
import data_processor as dp
import mychen76_donut as mcd


# Enum to select the NER method
class NerMethod(Enum):
    mistral_llm = 1
    donut = 2
    mychen_donut =3

# Initialize DonutRunner with the model path from data_processor
donut_runner = donut.DonutRunner(dp.path_model)
mc_donut_runner = mcd.MychenDonutTestRunner(dp.path_model)

# Wrapper function to run NER based on selected method
def run_ner(image, text: str, path, ner_method: NerMethod = NerMethod.mistral_llm):
    if ner_method == NerMethod.mistral_llm:
        llm_params = lllm.LlmParameters(path)
        return lllm.ask_local_llm(llm_params, text)
    elif ner_method == NerMethod.donut:
        return donut_runner.run_donut(image)
    elif ner_method == NerMethod.mychen_donut:
        return mc_donut_runner.run_donut(image, text)
    else:
        return []