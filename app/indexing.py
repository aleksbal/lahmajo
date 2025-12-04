# rag_ollama_demo/indexing.py
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings


def build_vector_store() -> InMemoryVectorStore:
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        header_template={"User-Agent": "Mozilla/5.0"},
        bs_kwargs={"parse_only": bs4_strainer},
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    # Embeddings via local Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = InMemoryVectorStore(embeddings)

    # 🔴 This is missing in your snippet: actually index the splits
    vector_store.add_documents(documents=all_splits)

    return vector_store
