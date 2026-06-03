3.June.2026

What is RAG?
-retrival of info from the given pdf 
What is an embedding?
-vector values
Why not send the entire PDF to an LLM?
-consuming to break down the pdf as embedding and chunking help breaking down the pdf , so llm can perform better
What is a vector database?
-values as vector in db
What is semantic search?
- meaning based search 

14.09
What is chunking?

Breaking large documents into smaller pieces before embedding.

Why?

- Better retrieval
- Lower token usage
- Better semantic matching

What is overlap?

Repeating part of the previous chunk in the next chunk.

Example:

Chunk 1:
A B C D E

Chunk 2:
D E F G H

Purpose:
Prevent context loss at boundaries.