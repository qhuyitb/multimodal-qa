import pdfplumber
import docx
import os
import pathlib
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P 
from docx.oxml.table import CT_Tbl
from utils.helpers import get_data_dir, get_project_root

output_base_dir = get_data_dir("output/documents")
os.makedirs(output_base_dir, exist_ok=True)

def extract_pdf_text(pdf_path, output_dir=None):
    """Trích xuất text từ PDF file"""
    output_dir = pathlib.Path(output_dir or output_base_dir) / "pdf"
    os.makedirs(output_dir, exist_ok=True)
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                text += f"\n[table]\n{_format_table(table)}[/table]\n\n"
            if page_text := page.extract_text():
                text += page_text + "\n"

    txt_path = output_dir / f"{pathlib.Path(pdf_path).stem}.txt"
    txt_path.write_text(text, encoding="utf-8")
    return text

def extract_docx_text(docx_path, output_dir=None):
    """Trích xuất text từ DOCX file"""
    output_dir = pathlib.Path(output_dir or output_base_dir) / "docx"
    os.makedirs(output_dir, exist_ok=True)
    
    doc = docx.Document(docx_path)
    text = ""
    for element in doc.element.body:
        if isinstance(element, CT_P):
            text += Paragraph(element, doc).text + "\n"
        elif isinstance(element, CT_Tbl):
            text += f"\n[TABLE]\n{_extract_table_from_docx(Table(element, doc))}[/TABLE]\n\n"
    
    txt_path = output_dir / f"{pathlib.Path(docx_path).stem}.txt"
    txt_path.write_text(text, encoding="utf-8")
    return text


def _format_table(table):
    if not table:
        return ""
    col_widths = [max(len(str(row[col])) for row in table) for col in range(len(table[0]))]
    
    separator = "-" + "-".join("-" * (w + 2) for w in col_widths) + "-\n"
    
    result = separator
    for row in table:
        result += "|"
        for col_idx, cell in enumerate(row):
            cell_text = str(cell or "")
            result += f" {cell_text:<{col_widths[col_idx]}} |"
        result += "\n" + separator
    return result


def _extract_table_from_docx(table):
    if not table:
        return ""
    data = []
    for row in table.rows:
        row_data =[cell.text for cell in row.cells]
        data.append(row_data)
    return _format_table(data)


class DocumentExtractor:
    """Extractor cho các loại tài liệu (PDF, DOCX)"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or output_base_dir
    
    def extract(self, document_path):
        document_path = pathlib.Path(document_path)
        if document_path.suffix.lower() == '.pdf':
            return extract_pdf_text(document_path, self.output_dir)
        elif document_path.suffix.lower() in ['.docx', '.doc']:
            return extract_docx_text(document_path, self.output_dir)
        else:
            raise ValueError(f"Unsupported file type: {document_path.suffix}")
    
    def extract_pdf(self, pdf_path, output_dir=None):
        return extract_pdf_text(pdf_path, output_dir)
    
    def extract_docx(self, docx_path, output_dir=None):
        return extract_docx_text(docx_path, output_dir)


if __name__ == "__main__":
    # sample_pdf = project_root / "data/input/documents/pdf/demo_pdf.pdf"
    # sample_docx = project_root / "data/input/documents/docx/demo_docx.docx"

    # pdf_text = extract_pdf_text(sample_pdf, output_base_dir / "pdf")
    # print("Extracted PDF Text:")
    # print(pdf_text)

    # docx_text = extract_docx_text(sample_docx, output_base_dir / "docx")
    # print("Extracted DOCX Text:")
    # print(docx_text)
    pass
    