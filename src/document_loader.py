from pypdf import PdfReader


def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text
        })

    return pages