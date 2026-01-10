# Document extraction (PDF, DOCX)
import pdfplumber
import docx
import os
import pathlib
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P 
from docx.oxml.table import CT_Tbl


project_root = pathlib.Path(__file__).parent.parent.parent
output_base_dir = project_root / "data" / "output" / "documents"
os.makedirs(output_base_dir, exist_ok=True)

def extract_pdf_text(pdf_path, output_dir=None):
    if output_dir is None:
        output_dir = output_base_dir / "pdf"
    else:
        output_dir = pathlib.Path(output_dir) / "pdf"
    
    os.makedirs(output_dir, exist_ok=True)
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    text += "\n[table]\n"
                    text += _format_table(table)
                    text += "[/table]\n\n"
            
           
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    name = pathlib.Path(pdf_path).stem
    txt_path = f"{output_dir}/{name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

def extract_docx_text(docx_path, output_dir=None):
    """Extract text from DOCX file
    
    Args:
        docx_path: Path to DOCX file
        output_dir: Optional output directory. If None, uses default data/output/documents/docx/
    
    Returns:
        Extracted text
    """
    if output_dir is None:
        output_dir = output_base_dir / "docx"
    else:
        output_dir = pathlib.Path(output_dir) / "docx"
    
    os.makedirs(output_dir, exist_ok=True)
    
    doc = docx.Document(docx_path)
    text = ""
    for element in doc.element.body:
        if isinstance(element, CT_P):
            para = Paragraph(element, doc)
            text += para.text + "\n"
        elif isinstance(element, CT_Tbl):
            table = Table(element, doc)
            text += "\n[TABLE]\n"
            text += _extract_table_from_docx(table)
            text += "[/TABLE]\n\n"
    
    name = pathlib.Path(docx_path).stem
    txt_path = f"{output_dir}/{name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return text


# Format table thành text đẹp
def _format_table(table):
    if not table:
        return ""
    col_widths = []
    for col_idx in range(len(table[0])):
        max_width = max(len(str(row[col_idx])) for row in table)
        col_widths.append(max_width)
    
    # Vẽ border
    separator = "-" + "-".join("-" * (w + 2) for w in col_widths) + "-\n"
    
    # Format từng row
    result = separator
    for row in table:
        result += "|"
        for col_idx, cell in enumerate(row):
            cell_text = str(cell or "")
            result += f" {cell_text:<{col_widths[col_idx]}} |"
        result += "\n" + separator
    return result


# Format table từ DOCX thành text 
def _extract_table_from_docx(table):
    if not table:
        return ""
    data = []
    for row in table.rows:
        row_data =[cell.text for cell in row.cells]
        data.append(row_data)
    return _format_table(data)

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
    