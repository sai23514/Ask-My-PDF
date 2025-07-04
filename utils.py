import fitz
from typing import List

def extract_text_chunks(file, max_chunks: int = 4) -> List[str]:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    chunks = []
    for page in doc:
        text = page.get_text()
        if text:
            chunks.append(text)
        if len(chunks) >= max_chunks:
            break
    return chunks

def load_pdf(chunks: List[str], question: str) -> str:
    # Just concatenate for now. Improve with similarity search later.
    return "\n\n".join(chunks)
