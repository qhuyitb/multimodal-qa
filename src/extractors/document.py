# Document extraction (PDF, DOCX)
import pdfplumber
import docx
import os
import pathlib

output_dir = "/data/output/documents"
os.makedirs(output_dir, exist_ok=True)
def extract_pdf_text(pdf_path, output_dir):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        

    name = pathlib.Path(pdf_path).stem
    txt_path = f"{output_dir}/{name}.txt"
    with open(txt_path, "w") as f:
        f.write(text)
    return text

def extract_docx_text(docx_path, output_dir):
    
    doc = docx.Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    name =pathlib.Path(docx_path).stem
    txt_path = f"{output_dir}/{name}.txt"
    with open(txt_path, "w") as f:
        f.write(text)
    
    return text

if __name__ == "__main__":
    pass
    