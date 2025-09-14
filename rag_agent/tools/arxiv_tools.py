from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

# ArXiv Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(arxiv_wrapper=arxiv_wrapper)
arxiv.description = (
    "ArXiv tool. Use when the query references arXiv or includes an ArXiv ID."
)
