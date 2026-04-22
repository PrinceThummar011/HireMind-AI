from fpdf import FPDF

def create_pdf_from_text(text: str) -> bytes:
    """
    Creates a simple PDF from text using fpdf2.
    """
    # Standard fonts in FPDF only support latin-1. 
    # We replace common problematic unicode characters.
    text = text.replace('\u2013', '-').replace('\u2014', '-') # dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'") # single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"') # double quotes
    text = text.replace('\u2022', '*') # bullets
    
    # Encode to latin-1 and ignore anything else to prevent crashes
    text = text.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 10, txt=text)
    
    return bytes(pdf.output())
