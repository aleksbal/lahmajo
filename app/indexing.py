# app/indexing.py
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


def build_vector_store() -> InMemoryVectorStore:
    # 1) Load the raw document from the web
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        header_template={"User-Agent": "Mozilla/5.0"},  # avoids USER_AGENT warning
        bs_kwargs={"parse_only": bs4_strainer},
    )

    docs = loader.load()  # -> list[Document]

    # 2) Embeddings (we'll reuse same instance for chunking + vector store)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 3) Semantic chunker instead of RecursiveCharacterTextSplitter
    #    It detects semantic breakpoints using embeddings.
    text_splitter = SemanticChunker(
        embeddings,
        # Optional knobs to tweak:
        # breakpoint_threshold_type="percentile",
        # breakpoint_threshold_amount=95,
    )

    # SemanticChunker expects a list of strings or texts;
    # we take the page_content from each loaded Document.
    raw_texts = [d.page_content for d in docs]

    # This returns list[Document] again, but now *semantically* chunked.
    semantic_chunks = text_splitter.create_documents(raw_texts)

    # (Optional) Preserve a simple "source" metadata
    for chunk in semantic_chunks:
        chunk.metadata.setdefault("source", "lilianweng_agent_blog")

    # 4) Build vector store and add chunks
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(semantic_chunks)

    return vector_store

