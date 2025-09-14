# 🤖 RAG Agent for Multiple Data Sources

A powerful Retrieval-Augmented Generation (RAG) agent that can query multiple data sources to answer questions with high accuracy and context-awareness.

## ✨ Features

### 🔍 **Multi-Source Intelligence**
- **Wikipedia Quick Search**: Fast factual answers
- **Wikipedia Deep RAG**: Detailed context, timelines, and quotations
- **ArXiv**: Research papers and academic preprints
- **Web Search via Brave**: General web queries with intelligent scraping
- **LangSmith Documentation**: Specialized retriever for LangChain docs
- **Document RAG**: Index and query your own documents (PDF, DOCX, TXT, CSV, HTML)

### 🎯 **Smart Document Handling**
- **Auto-Discovery**: Automatically finds supported documents in `documents/` folder
- **Selective Indexing**: Choose exactly which documents to index (prevents context dilution)
- **Multiple Formats**: Supports PDF, DOCX, TXT, MD, CSV, HTML files
- **URL Support**: Index web pages directly

### 💻 **Dual Interface Options**
- **Streamlit Web App**: User-friendly web interface with visual document selection
- **CLI App**: Terminal-based interface for automation and scripting

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Optional: Brave Search API key for web search

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd rag_agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   BRAVE_SEARCH_API_KEY=your_brave_api_key_here  # Optional
   ```

## 🎮 Usage

### Option 1: Streamlit Web App (Recommended)

Launch the interactive web interface:

```bash
# From the rag_agent directory
streamlit run app.py
```

**Features:**
- Visual document selection from sidebar
- Real-time chat interface
- Conversation memory management
- Multi-document indexing with checkboxes
- URL indexing support

### Option 2: CLI App

For terminal-based usage:

```bash
# From the rag_agent directory
python cli_app.py
```

**Commands:**
- `:index <files>` - Index specific documents
- `:status` - Check indexing status
- `:reset` - Clear conversation memory
- `:exit` - Quit the application

## 📁 Project Structure

```
rag_agent/
├── app.py                 # Streamlit web application
├── cli_app.py            # CLI application (backup)
├── agent.py              # OpenAI tools agent setup
├── config.py             # Environment configuration
├── tool_registry.py      # Tool orchestration
├── main.py               # Example usage scripts
├── requirements.txt      # Python dependencies
├── documents/            # Place your documents here
│   ├── Neha_Gaonkar_Resume.txt
│   ├── deep-learning-guide.pdf
│   └── ...
└── tools/                # Modular tool implementations
    ├── document_tools.py # Document indexing & RAG
    ├── web_tools.py      # Web search & scraping
    ├── wikipedia_tools.py # Wikipedia integration
    ├── arxiv_tools.py    # Academic paper search
    └── langsmith_tools.py # LangChain docs
```

## 📚 Adding Your Documents

1. **Place files in the `documents/` directory:**
   ```
   rag_agent/documents/
   ├── my_resume.pdf
   ├── research_paper.docx
   ├── notes.txt
   └── company_handbook.html
   ```

2. **Supported formats:**
   - PDF (.pdf)
   - Word documents (.docx, .doc)
   - Text files (.txt, .md)
   - CSV files (.csv)
   - HTML files (.html, .htm)

3. **Index via Web App:**
   - Launch Streamlit app
   - Select documents in sidebar
   - Click "Index Selected"

4. **Index via CLI:**
   ```bash
   :index documents/my_resume.pdf,documents/research_paper.docx
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | Your OpenAI API key for GPT models |
| `BRAVE_SEARCH_API_KEY` | ❌ | Brave Search API for web queries |

### Customization

**Modify chunk size and overlap in `tools/document_tools.py`:**
```python
result = doc_index(
    files_csv="documents/your_file.pdf",
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200     # Adjust overlap
)
```

## 🎯 Example Queries

The agent can handle various types of questions:

### Document Questions
```
"What are Neha's technical skills?"
"Summarize the research findings in this paper"
"What are the main points in the company handbook?"
```

### Academic Research
```
"Find recent papers on transformer architectures"
"What are the latest developments in quantum computing?"
```

### General Knowledge
```
"Who won the Nobel Prize in Physics 2023?"
"What are the benefits of renewable energy?"
```

### Web Search
```
"What are the current trending topics on AI?"
"Latest news about climate change policies"
```

## 🛠️ Development

### Adding New Tools

1. **Create tool file in `tools/` directory:**
   ```python
   # tools/custom_tools.py
   from langchain.tools import StructuredTool

   def custom_search(query: str) -> str:
       # Your implementation
       return f"Results for: {query}"

   custom_tool = StructuredTool.from_function(
       func=custom_search,
       name="custom_search",
       description="Search custom data source"
   )
   ```

2. **Register in `tool_registry.py`:**
   ```python
   from tools.custom_tools import custom_tool

   tools = [custom_tool, ...]  # Add to existing tools
   ```

### Testing

Run example queries:
```bash
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for tool orchestration
- Powered by [OpenAI GPT](https://openai.com/) for intelligent responses
- Web search via [Brave Search API](https://brave.com/search/api/)
- UI powered by [Streamlit](https://streamlit.io/)

---

**Happy chatting with your RAG Agent! 🤖✨**
