# Retrieval Layer

## Purpose

The retrieval layer connects:

- Query
- Embedding generation
- Vector search

## Workflow

User Query
↓
Query Embedding
↓
FAISS Search
↓
Top-k Chunks
↓
Return Results

## Responsibilities

- Query embedding generation
- Similarity search
- Result ranking
- Metadata preservation

## Future Improvements

- Hybrid Search
- BM25 + Vector Search
- Reranking
- Metadata Filters