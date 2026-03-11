from langchain_community.llms import CTransformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
import os
import timeit
import json
import data_processor as dp

# JSON output parser to enforce structured output from the LLM
output_parser = JsonOutputParser()

class LlmParameters:
    """
    Class stores configuration parameters for the local LLM mistral
    """
    def __init__(self, path_model):
        self.max_new_tokens = 1048
        self.temperature = 0.00
        self.model_type = 'mistral'
        self.model_path = os.path.join(path_model, 'mistral-7b-instruct-v0.1.Q5_K_M.gguf')
        self.return_source_documents = True
        self.vector_count = 2
        self.chunk_size = 300
        self.chunk_overlap = 30
        self.embeddings_model = 'sentence-transformers/all-mpnet-base-v2'
        self.qa_template = """Use the following pieces of information to extract fields from invoice data.
Context: {context}
Fields to extract: {question}
Provide output only in valid JSON structure in format
\"field\": \"value extracted from the context\"
If can't find a field in the document context, just leave value empty in JSON format, don't try to make up an answer.
Do not add any additional words or symbols except valid JSON format itself so I can parse it easily. 
 """

def setup_llm(params: LlmParameters):
    return CTransformers(model=params.model_path,
                         model_type=params.model_type,
                         max_new_tokens=params.max_new_tokens,
                         temperature=params.temperature
                         )

# Build vector store for retrieval
def run_ingest(params: LlmParameters, text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.chunk_size,
                                                   chunk_overlap=params.chunk_overlap,
                                                   length_function=len,
                                                   is_separator_regex = False)
    documents = text_splitter.create_documents([text])
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=params.embeddings_model,
                                       model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Create a QA prompt template
def set_qa_prompt(params: LlmParameters):
    return PromptTemplate(template=params.qa_template, input_variables=['context', 'question'],
                          partial_variables={"format_instruction": output_parser.get_format_instructions()})


def build_retrieval_qa_chain(llm, prompt, vectorstore, params: LlmParameters):
    retriever = vectorstore.as_retriever(search_kwargs={'k': params.vector_count})

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=retriever,
                                           return_source_documents=params.return_source_documents,
                                           chain_type_kwargs={'prompt': prompt})
    return qa_chain


# Setup QA chain with LLM, vectorstore, and prompt
def setup_qa_chain(vectorstore, params: LlmParameters):
    llm = setup_llm(params)
    qa_prompt = set_qa_prompt(params)
    qa_chain = build_retrieval_qa_chain(llm, qa_prompt, vectorstore, params)

    return qa_chain

# Run the full LLM extraction pipeline
def run_llm(llm_params: LlmParameters, text: str):
    start = timeit.default_timer()
    vectors = run_ingest(llm_params, text)

    query_input = 'Invoice number, Invoice date, Shipping cost, Tax, Total amount'

    qa_chain = setup_qa_chain(vectors, llm_params)
    response = qa_chain({'query': query_input})
    end = timeit.default_timer()
    print('LLM time spent ' + str(end - start))

    result = response["result"]
    result_extracted = dp.extract_json_substrings(result)

    return result_extracted

# Wrapper to repeatedly attempt LLM extraction if it fails
def ask_local_llm(llm_params: LlmParameters, text: str):
    valid_result = False
    try_num = 0

    while not valid_result:
        try_num = try_num + 1
        print('LLM try ' + str(try_num))

        try:
            result_extracted = run_llm(llm_params, text)
            print('Found entities: ')
            print(result_extracted)
            return result_extracted
        except ValueError as e:
            print(e)
            print("LLM try failed")

        if try_num > 3:
            break

    print("LLM can't find an answer :(")
    return json.dumps({})