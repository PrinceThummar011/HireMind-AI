import fitz


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_chunks: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page in document:
            text_chunks.append(page.get_text("text"))
    return "\n".join(text_chunks).strip()


def extract_text_from_uploaded_pdf(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    if not pdf_bytes:
        return ""
    return extract_text_from_pdf_bytes(pdf_bytes)
