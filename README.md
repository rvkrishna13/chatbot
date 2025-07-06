# Wikipedia Chatbot with Gemini AI

A conversational AI chatbot that can answer questions about Wikipedia pages using Google's Gemini AI models and LlamaIndex for document processing.

## Features

- 🤖 **AI-Powered Chat**: Uses Google's Gemini models (1.5-flash, 1.5-pro, 1.0-pro) for intelligent responses
- 📚 **Wikipedia Integration**: Automatically indexes and searches Wikipedia pages
- 🔍 **Semantic Search**: Uses HuggingFace embeddings for accurate document retrieval
- 🛠️ **ReAct Agent**: Implements reasoning and action capabilities for better responses
- 🌐 **Web Interface**: Beautiful Chainlit-based chat interface
- ⚙️ **Configurable**: Choose different Gemini models and Wikipedia pages

## Architecture

- **Frontend**: Chainlit web interface
- **AI Models**: Google Gemini (via LlamaIndex adapter)
- **Document Processing**: LlamaIndex with Wikipedia reader
- **Embeddings**: HuggingFace BGE-small-en model
- **Vector Database**: In-memory vector store with LlamaIndex

## Prerequisites

- Python 3.9+
- Google Gemini API key
- Internet connection for Wikipedia access

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv chatbot_env
   source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key:**
   Create an `apikeys.yml` file in the project root:
   ```yaml
   gemini:
     api_key: 'your-gemini-api-key-here'
   ```

## Usage

1. **Start the chatbot:**
   ```bash
   chainlit run chat_agent.py
   ```

2. **Configure the chatbot:**
   - Open the web interface (usually at http://localhost:8000)
   - Click the settings icon (⚙️) at the bottom left
   - Select a Gemini model (recommended: gemini-1.5-flash)
   - Enter Wikipedia pages to index (e.g., "please index: Paris, London, Tokyo")

3. **Start chatting:**
   - Ask questions about the indexed Wikipedia pages
   - The chatbot will search through the indexed content and provide answers

## Example Usage

```
User: please index: Paris, France, Eiffel Tower
Bot: ✅ Wikipage(s) 'Paris, France, Eiffel Tower' successfully indexed with 3 documents. You can now ask questions!

User: What is the Eiffel Tower made of?
Bot: The Eiffel Tower is made primarily of wrought iron (puddled iron). The structure consists of approximately 7,300 tons of iron, which was prefabricated in the Eiffel factory in Levallois-Perret, a suburb of Paris...
```

## Project Structure

```
chatbot/
├── chat_agent.py          # Main chatbot application
├── index_wikipages.py     # Wikipedia indexing logic
├── utils.py              # Utility functions (API key management)
├── apikeys.yml           # API key configuration (create this)
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .chainlit/           # Chainlit configuration
```

## Key Components

### `chat_agent.py`
- Main Chainlit application
- ReAct agent implementation
- Chat interface and settings management

### `index_wikipages.py`
- Wikipedia page parsing and indexing
- Document processing with LlamaIndex
- Vector store creation

### `utils.py`
- API key management from YAML file
- Configuration utilities

## Configuration

### Available Models
- `gemini-1.5-flash`: Fast, efficient, recommended for most use cases
- `gemini-1.5-pro`: More capable, better reasoning
- `gemini-1.0-pro`: Stable, well-tested

### Wikipedia Page Format
- Use the format: "please index: Page1, Page2, Page3"
- Pages are automatically parsed and indexed
- Supports multiple pages in a single request

## Troubleshooting

### Common Issues

1. **"Agent is not available"**
   - Make sure you've configured the settings (model and Wikipedia pages)
   - Check that the Wikipedia pages exist

2. **"Failed to create index"**
   - Verify your Gemini API key is valid
   - Check internet connection for Wikipedia access
   - Ensure Wikipedia page names are correct

3. **Model errors**
   - Try switching to a different Gemini model
   - Check your API quota and rate limits

### Debug Mode
The application includes extensive logging. Check the terminal output for detailed error messages and debugging information.

## Dependencies

Key dependencies include:
- `chainlit`: Web interface
- `llama-index`: Document processing and RAG
- `google-generativeai`: Gemini AI integration
- `llama-index-llms-gemini`: Gemini LLM adapter
- `llama-index-readers-wikipedia`: Wikipedia integration
- `llama-index-embeddings-huggingface`: Embedding models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the terminal logs for error messages
3. Open an issue on the repository

---

**Note**: This chatbot requires a valid Google Gemini API key. Make sure to keep your API key secure and never commit it to version control.
