from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.agent.react.base import ReActAgent                   
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.llms.openai.base import OpenAI
import chainlit as cl
from chainlit.input_widget import Select, TextInput
from index_wikipages import create_index
from utils import get_apikey
import google.generativeai as genai

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
                values=['gemini-2.0-flash'],
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
    query_engine_tools = [
        QueryEngineTool(
                query_engine=wikisearch_engine(index),
                metadata=ToolMetadata(
                    name='Wikipedia', 
                    description="Useful for performing searches on the wikipedia knowledge base"
                ),
            )
    ]

    geminikey = get_apikey()

    genai.configure(api_key=geminikey)
    llm = genai.GenerativeModel(MODEL)
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
    query = settings['WikiPageRequest']
    if not isinstance(query, str):
        query = str(query)
    index = create_index(query)

    print("on_settings_update", settings)
    MODEL = settings['MODEL']
    if not isinstance(MODEL, str):
        MODEL = str(MODEL)
    agent = create_react_agent(MODEL)
    await cl.Message(
        author="Agent", content=f"""Wikipage(s) "{query}" successfully indexed"""
    ).send()


@cl.on_message
async def main(message: str):
    global agent
    if agent:
        print("Agent is available, processing message.")
    print("Received message:", message)
    if not isinstance(message, str):
        message = str(message)
    if agent:
        print("Agent is available, processing message.")
        response = await cl.make_async(agent.chat)(message)
        await cl.Message(author="Agent", content=response.response if isinstance(response, AgentChatResponse) else str(response)).send()
    else:
        print("Agent is not available")
