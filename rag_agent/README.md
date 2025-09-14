# RAG Agent for Multiple Data Sources

This is a Python implementation of a Retrieval-Augmented Generation (RAG) agent that can query multiple data sources to answer questions.

## Features

- **Wikipedia Quick Search**: For short factual questions.
- **Wikipedia Deep RAG**: For detailed context, timelines, and quotations.
- **ArXiv**: For research papers and preprints.
- **Web Search via Brave**: General web queries with scraping and RAG.
- **LangSmith Documentation**: Specific retriever for LangChain docs.
- **Document RAG**: Index and query user-provided documents (PDF, DOCX, TXT, CSV, HTML, URLs).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   BRAVE_SEARCH_API_KEY=your_brave_api_key
   ```

## Usage

Run the main script to see example queries:
```bash
python main.py
```

## Structure

- `config.py`: Environment setup and API keys.
- `tools.py`: Definitions of all RAG tools.
- `agent.py`: Setup of the OpenAI tools agent and executor.
- `main.py`: Example runs with various queries.
