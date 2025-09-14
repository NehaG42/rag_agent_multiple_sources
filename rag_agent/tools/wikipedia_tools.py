from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import StructuredTool

# Wikipedia Quick Search
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki.description = (
    "Wikipedia quick lookup. Use for short factual questions that need a brief summary. "
    "Do NOT use for detailed/ambiguous multi-part queries or when quotes/timelines are needed; "
    "prefer wikipedia_rag in those cases."
)

# Wikipedia RAG
def wikipedia_rag(query: str, top_docs: int = 3) -> str:
    try:
        wiki_docs = WikipediaLoader(query=query, load_max_docs=top_docs).load()
    except Exception as e:
        return f"Failed to load from Wikipedia: {e}"

    if not wiki_docs:
        return "No Wikipedia pages found."

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(wiki_docs)

    vdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    rel_docs = vdb.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(query)

    if not rel_docs:
        return "No relevant passages were found in the fetched Wikipedia pages."

    out, seen = [], []
    for d in rel_docs:
        title = d.metadata.get("title") or "Wikipedia"
        if title not in seen:
            seen.append(title)
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + " ..."
        url = d.metadata.get("source") or d.metadata.get("wikipedia_url") or ""
        src = f"{title} — {url}" if url else title
        out.append(f"- Excerpt: {text}\n  Source: {src}")

    return (
        "Wikipedia RAG answer (supporting snippets):\n"
        + "\n".join(out[:4])
        + "\n\nPages:\n"
        + "\n".join(f"* {t}" for t in seen[:5])
    )

wikipedia_rag_tool = StructuredTool.from_function(
    func=wikipedia_rag,
    name="wikipedia_rag",
    description=(
        "Deep Wikipedia RAG. Use when the question needs detailed context, comparisons, timelines, or quotations "
        "from Wikipedia beyond a short summary. Args: query:str, top_docs:int=3"
    ),
)
