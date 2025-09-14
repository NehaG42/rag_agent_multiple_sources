from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, BSHTMLLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import StructuredTool

class DocumentState:
    def __init__(self):
        self.vdb = None
        self.chunk_count = 0
        self.indexed = False

doc_state = DocumentState()

def doc_index(files_csv: str = "./NehaGaonkarDataScientist.pdf", urls_csv: str = "", chunk_size: int = 1200, chunk_overlap: int = 200) -> str:
    """
    Ingest user-provided documents into an in-memory FAISS index.
    - files_csv: comma-separated local file paths (pdf, docx, txt, md, csv, html)
    - urls_csv: comma-separated URLs to fetch and index
    """
    paths = [p.strip() for p in files_csv.split(",") if p.strip()]
    urls = [u.strip() for u in urls_csv.split(",") if u.strip()]

    docs = []

    # Load local files
    for p in paths:
        try:
            lo = None
            pl = p.lower()
            if pl.endswith(".pdf"):
                lo = PyPDFLoader(p)
            elif pl.endswith(".docx") or pl.endswith(".doc"):
                lo = Docx2txtLoader(p)
            elif pl.endswith(".csv"):
                lo = CSVLoader(p)
            elif pl.endswith(".html") or pl.endswith(".htm"):
                lo = BSHTMLLoader(p)
            elif pl.endswith(".txt") or pl.endswith(".md"):
                lo = TextLoader(p, encoding="utf-8")
            else:
                # skip unknown types silently to avoid heavy 'unstructured' dependency
                continue
            docs.extend(lo.load())
        except Exception:
            continue

    # Load URLs
    for url in urls:
        try:
            page_docs = WebBaseLoader(url).load()
            for pd in page_docs:
                pd.metadata["source"] = url
            docs.extend(page_docs)
        except Exception:
            continue

    if not docs:
        return "No documents loaded. Provide valid paths in files_csv or URLs in urls_csv."

    # Chunk and (create or update) vectorstore
    splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    chunks = splitter.split_documents(docs)

    if doc_state.vdb is None:
        doc_state.vdb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    else:
        doc_state.vdb.add_documents(chunks)

    doc_state.chunk_count += len(chunks)
    doc_state.indexed = doc_state.vdb is not None and doc_state.chunk_count > 0

    return f"Indexed {len(chunks)} chunks from {len(paths)} file(s) and {len(urls)} URL(s)."

def doc_rag_search(query: str, k: int = 4) -> str:
    """
    Query the user-provided document corpus previously indexed with doc_index.
    """
    if doc_state.vdb is None:
        return "No indexed documents. Run doc_index first with files_csv and/or urls_csv."
    rel = doc_state.vdb.as_retriever(search_kwargs={"k": int(k)}).get_relevant_documents(query)
    if not rel:
        return "No relevant passages found in the indexed documents."

    out, seen = [], []
    for d in rel:
        src = d.metadata.get("source") or d.metadata.get("file_path") or d.metadata.get("source_path") or "uploaded_document"
        if src not in seen:
            seen.append(src)
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > 700:
            text = text[:700] + " ..."
        out.append(f"- Excerpt: {text}\n  Source: {src}")
    return "Document RAG answer (supporting snippets):\n" + "\n".join(out[:4]) + "\n\nSources:\n" + "\n".join(f"* {s}" for s in seen[:8])

doc_index_tool = StructuredTool.from_function(
    func=doc_index,
    name="doc_index",
    description=("Ingest user documents into a persistent (session) vector index. "
                 "Use when the user uploads or provides links. "
                 "Args: files_csv:str (comma-separated file paths), urls_csv:str (comma-separated URLs), "
                 "chunk_size:int=1200, chunk_overlap:int=200"),
)
doc_rag_tool = StructuredTool.from_function(
    func=doc_rag_search,
    name="doc_rag_search",
    description=("Query the user's indexed documents (from doc_index). "
                 "Use for questions that refer to the uploaded/added documents. "
                 "Args: query:str, k:int=4"),
)
