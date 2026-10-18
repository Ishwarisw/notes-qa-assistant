from pypdf import PdfReader

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

if __name__ == "__main__":
    text = read_pdf("sample.pdf")
    print("Total characters:", len(text))
    print(text[:500])
