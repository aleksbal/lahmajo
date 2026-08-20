# Embeddings Objects Explanation

## Overview

There are two embedding model instances in the codebase, each with a distinct purpose:

1. **`semantic_chunker_embeddings`** - Used for semantic chunking analysis
2. **`document_embedding_model`** - Used for embedding documents into the vector store

## Purpose of Each

### `semantic_chunker_embeddings`
- **Purpose**: Used by `SemanticChunker` to analyze text and find semantic breakpoints
- **When used**: Only when `use_semantic=True` (semantic chunking strategy)
- **What it does**: 
  - Analyzes text to determine where to split documents based on semantic meaning
  - Makes many embedding calls during the chunking process to find optimal split points
- **Why separate instance**: 
  - SemanticChunker uses embeddings heavily during analysis
  - Keeps the analysis process isolated from document storage embeddings

### `document_embedding_model`
- **Purpose**: Used to embed document chunks that will be stored in the vector database
- **When used**: Always - to create the vector store and embed all chunks
- **What it does**: 
  - Converts text chunks into vector embeddings for similarity search
  - Used by InMemoryVectorStore to embed and store documents
- **Why separate instance**: 
  - Keeps the embedding model clean and dedicated to document storage
  - Prevents any potential state issues from chunking operations

## Why Two Separate Instances?

Even though both use the same model (`nomic-embed-text`), they're kept separate because:
1. **Separation of concerns**: Chunking analysis vs. document storage
2. **Defensive programming**: Prevents any potential state contamination
3. **Clarity**: Makes it explicit which embeddings are used for what purpose

## Naming Rationale

The new names are clearer:
- **`semantic_chunker_embeddings`**: Clearly indicates it's only for semantic chunking analysis
- **`document_embedding_model`**: Clearly indicates it's for embedding documents for storage

This makes the code self-documenting and removes confusion about when each is used.
