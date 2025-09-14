from langchain_community.document_loaders import WebBaseLoader, BraveSearchLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import StructuredTool
from config import BRAVE_API_KEY

def web_rag_search(query: str, count: int = 5) -> str:
    if not BRAVE_API_KEY:
        return "BRAVE_SEARCH_API_KEY is not set."
    count = max(1, min(int(count), 10))

    # 1) Search
    serp_docs = BraveSearchLoader(query=query, api_key=BRAVE_API_KEY, search_kwargs={"count": count}).load()

    # 2) Links
    links = []
    for d in serp_docs:
        link = d.metadata.get("link")
        if link and link not in links:
            links.append(link)
    if not links:
        return "No web results found."

    # 3) Scrape
    web_docs = []
    for url in links:
        try:
            for pd in WebBaseLoader(url).load():
                pd.metadata["source"] = url
                web_docs.append(pd)
        except Exception:
            pass
    if not web_docs:
        return "Failed to load content from the top web results."

    # 4) Chunk → 5) Embed → 6) Similarity search
    chunks = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(web_docs)
    vdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    rel_docs = vdb.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(query)

    if not rel_docs:
        return "No relevant passages were found in the crawled pages."

    # 7) Compose short answer with sources
    out, seen = [], []
    for d in rel_docs:
        src = d.metadata.get("source") or d.metadata.get("link") or "unknown"
        if src not in seen:
            seen.append(src)
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + " ..."
        out.append(f"- Excerpt: {text}\n  Source: {src}")

    return (
        "Web RAG answer (supporting snippets):\n"
        + "\n".join(out[:4])
        + "\n\nTop sources:\n"
        + "\n".join(f"* {s}" for s in seen[:5])
    )

web_rag_tool = StructuredTool.from_function(
    func=web_rag_search,
    name="web_rag_search",
    description="General web RAG. Use when the question is NOT about Wikipedia, Arxiv, or LangSmith. Args: query:str, count:int=5",
)
