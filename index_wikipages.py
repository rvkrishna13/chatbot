from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.core.settings import Settings
from pydantic import BaseModel
from utils import get_apikey
import ast
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set Gemini as the default LLM for LlamaIndex - using a more reliable model
gemini_llm = Gemini(
    model_name="gemini-1.5-flash",  # Changed to a more reliable model
    api_key=get_apikey(),
    temperature=0.1
)
Settings.llm = gemini_llm

# define the data model in pydantic
class WikiPageList(BaseModel):
    "Data model for WikiPageList"
    pages: list


def wikipage_list(query):
    # Use the LlamaIndex Gemini LLM adapter
    llm = Gemini(
        model_name='gemini-1.5-flash',  # Changed to a more reliable model
        api_key=get_apikey(),
        temperature=0.1
    )

    prompt_template = """
    Given the input: "{query}", extract the Wikipedia page names mentioned after the phrase
    'please index:' and return them strictly as a Python list of strings.
    If there is only one page, return it inside a single-element list.
    Only return the list itself, nothing else.
    In any case i dont want the python code i only want the output as a list of strings so that i can use ast 
    to get the list to use in my task
    """

    try:
        response = llm.complete(prompt_template.format(query=query.lower()))
        print(f"LLM Response: {response.text}")
        
        # Clean the response text
        cleaned_text = response.text.strip()
        if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
            wikipage_requests = WikiPageList(pages=ast.literal_eval(cleaned_text))
        else:
            # Try to extract list from the response
            import re
            list_match = re.search(r'\[.*?\]', cleaned_text)
            if list_match:
                wikipage_requests = WikiPageList(pages=ast.literal_eval(list_match.group()))
            else:
                print("Could not parse list from response, using fallback")
                # Fallback: split by comma and clean
                pages = [page.strip() for page in query.replace('please index:', '').split(',') if page.strip()]
                wikipage_requests = WikiPageList(pages=pages)
        
        print(f"Parsed pages: {wikipage_requests.pages}")
        return wikipage_requests
        
    except Exception as e:
        print(f"Parsing Error: {e}")
        # Fallback: extract pages directly from query
        try:
            if 'please index:' in query.lower():
                pages_text = query.lower().split('please index:')[1].strip()
                pages = [page.strip() for page in pages_text.split(',') if page.strip()]
                return WikiPageList(pages=pages)
            else:
                return WikiPageList(pages=[])
        except:
            return WikiPageList(pages=[])


def create_wikidocs(wikipage_requests):
    try:
        reader = WikipediaReader()
        print(f"Loading Wikipedia pages: {wikipage_requests.pages}")
        documents = reader.load_data(pages=wikipage_requests.pages)
        print(f"Loaded {len(documents)} documents")
        return documents
    except Exception as e:
        print(f"Error loading Wikipedia documents: {e}")
        return []


def create_index(query):
    try:
        print(f"Creating index for query: {query}")
        wikipedia_requests = wikipage_list(query)
        
        if not wikipedia_requests.pages:
            print("No Wikipedia pages found!")
            return None
            
        documents = create_wikidocs(wikipedia_requests)
        
        if not documents:
            print("No documents loaded!")
            return None
            
        text_splits = SentenceSplitter(chunk_size=150, chunk_overlap=45)
        nodes = text_splits.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} nodes")
        
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        
        print(f"Index created successfully with {len(index.docstore.docs)} documents")
        return index
        
    except Exception as e:
        print(f"Error creating index: {e}")
        return None


if __name__ == "__main__":
    query = "please index: paris, lagos, lao"
    index = create_index(query)
    if index:
        print("INDEX CREATED SUCCESSFULLY", index)
    else:
        print("FAILED TO CREATE INDEX")
