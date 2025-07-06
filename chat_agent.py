from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.agent.react.base import ReActAgent                   
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.llms.gemini import Gemini
from llama_index.core.settings import Settings
import chainlit as cl
from chainlit.input_widget import Select, TextInput
from index_wikipages import create_index
from utils import get_apikey

# Set Gemini as the default LLM for LlamaIndex - using a more reliable model
gemini_llm = Gemini(
    model_name="gemini-1.5-flash",  # Changed to a more reliable model
    api_key=get_apikey(),
    temperature=0.1
)
Settings.llm = gemini_llm

index = None
agent = None

@cl.on_chat_start
async def on_chat_start():
    global index
    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="MODEL",
                label="Gemini - Model",
                values=['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro'],  # Updated model list
                initial_index=0,
            ),
            
            TextInput(id="WikiPageRequest", label="Request Wikipage")
        ]
    ).send()


def wikisearch_engine(index):
    query_engine = index.as_query_engine(
        response_mode="compact", verbose=True, similarity_top_k=10
    )
    return query_engine


def create_react_agent(MODEL):
    global index
    
    if index is None:
        print("ERROR: Index is None! Cannot create agent.")
        return None
        
    print(f"Creating agent with model: {MODEL}")
    print(f"Index has {len(index.docstore.docs)} documents")
    
    # Test the model first
    try:
        test_llm = Gemini(
            model_name=MODEL,
            api_key=get_apikey(),
            temperature=0.1
        )
        test_response = test_llm.complete("Hello, this is a test.")
        print(f"Model test successful: {test_response.text[:100]}...")
    except Exception as e:
        print(f"Model test failed: {e}")
        return None
    
    query_engine_tools = [
        QueryEngineTool(
                query_engine=wikisearch_engine(index),
                metadata=ToolMetadata(
                    name='Wikipedia', 
                    description="Useful for performing searches on the wikipedia knowledge base"
                ),
            )
    ]

    # Use LlamaIndex's Gemini LLM adapter instead of direct GenerativeModel
    llm = Gemini(
        model_name=MODEL,
        api_key=get_apikey(),
        temperature=0.1
    )
    
    agent = ReActAgent.from_tools(
        tools = query_engine_tools,
        llm=llm,
        verbose=True
    )
    return agent


@cl.on_settings_update
async def setup_agent(settings):
    global agent
    global index
    
    try:
        query = settings['WikiPageRequest']
        if not isinstance(query, str):
            query = str(query)
            
        print(f"Creating index for query: {query}")
        index = create_index(query)
        
        if index is None or len(index.docstore.docs) == 0:
            await cl.Message(
                author="Agent", 
                content=f"❌ Failed to create index for '{query}'. Please check if the Wikipedia pages exist and try again."
            ).send()
            return

        print("on_settings_update", settings)
        MODEL = settings['MODEL']
        if not isinstance(MODEL, str):
            MODEL = str(MODEL)
            
        agent = create_react_agent(MODEL)
        
        if agent is None:
            await cl.Message(
                author="Agent", 
                content=f"❌ Failed to create agent. Please try again."
            ).send()
            return
            
        await cl.Message(
            author="Agent", 
            content=f"✅ Wikipage(s) '{query}' successfully indexed with {len(index.docstore.docs)} documents. You can now ask questions!"
        ).send()
        
    except Exception as e:
        print(f"Error in setup_agent: {e}")
        await cl.Message(
            author="Agent", 
            content=f"❌ Error setting up agent: {str(e)}"
        ).send()


@cl.on_message
async def main(message: str):
    global agent
    print("Received message:", message)
    
    if not isinstance(message, str):
        message = str(message)
        
    if agent:
        print("Agent is available, processing message.")
        try:
            response = await cl.make_async(agent.chat)(message)
            response_text = response.response if isinstance(response, AgentChatResponse) else str(response)
            await cl.Message(author="Agent", content=response_text).send()
        except Exception as e:
            print(f"Error in agent chat: {e}")
            await cl.Message(
                author="Agent", 
                content=f"❌ Error processing your message: {str(e)}"
            ).send()
    else:
        print("Agent is not available")
        await cl.Message(
            author="Agent", 
            content="❌ Agent is not available. Please configure the settings first (select a model and enter Wikipedia pages to index)."
        ).send()
