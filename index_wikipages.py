from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OpenAIPydanticProgram
import google.generativeai as genai
from pydantic import BaseModel
from utils import get_apikey
import ast
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# define the data model in pydantic
class WikiPageList(BaseModel):
    "Data model for WikiPageList"
    pages: list


def wikipage_list(query):
    genai.configure(api_key=get_apikey())
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt_template = """
    Given the input: "{query}", extract the Wikipedia page names mentioned after the phrase
    'please index:' and return them strictly as a Python list of strings.
    If there is only one page, return it inside a single-element list.
    Only return the list itself, nothing else.
    In any case i dont want the python code i only want the output as a list of strings so that i can use ast 
    to get the list to use in my task
    """

    response = model.generate_content(prompt_template.format(query=query.lower()))
    print(response.text)
    try:
        wikipage_requests = WikiPageList(pages=ast.literal_eval(response.text.strip()))
        return wikipage_requests
    except Exception as e:
        print("Parsing Error:", e)
        return WikiPageList(pages=[])


def create_wikidocs(wikipage_requests):
    reader = WikipediaReader()
    documents = reader.load_data(pages=wikipage_requests)
    
    return documents


def create_index(query):
    global index
    wikipedia_requests = wikipage_list(query)
    documents = create_wikidocs(wikipedia_requests)
    text_splits = SentenceSplitter(chunk_size=150,chunk_overlap=45)
    nodes = text_splits.get_nodes_from_documents(documents)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    return index


if __name__ == "__main__":
    query = "/get wikipages: paris, lagos, lao"
    index = create_index(query)
    wikipage_list(query)
    print("INDEX CREATED", index)
