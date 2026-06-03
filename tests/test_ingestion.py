from src.ingestion.pdf_loader import PDFLoader

loader = PDFLoader()

document = loader.load("data/attention-is-all-you-need-Paper.pdf")

print("Pages:", document.page_count)
print("\nFirst 500 chars:\n")
print(document.full_text[:500])