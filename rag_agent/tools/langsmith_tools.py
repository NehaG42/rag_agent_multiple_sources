from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# LangSmith Retriever
loader = WebBaseLoader('https://docs.smith.langchain.com/')
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "FOR LANGSMITH ONLY. If the query mentions LangSmith or docs.smith.langchain.com, "
    "you MUST use this tool and no other."
)
