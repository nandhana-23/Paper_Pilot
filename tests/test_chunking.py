from src.ingestion.pdf_loader import PDFLoader
from src.chunking.text_chunker import TextChunker

document = PDFLoader().load(
    "data/attention-is-all-you-need-Paper.pdf"
)

chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

chunks = chunker.chunk(document)

print(f"Total chunks: {len(chunks)}")

print("\nFirst chunk:\n")
print(chunks[0].text)

print("\nMetadata:\n")
print(chunks[0].page_number)
print(chunks[0].chunk_id)