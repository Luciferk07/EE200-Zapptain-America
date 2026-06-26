import PyPDF2

pdf_path = r"C:\Users\karti\Downloads\EE200_course_project_summer_2026.pdf"
reader = PyPDF2.PdfReader(pdf_path)

output_path = r"c:\Users\karti\OneDrive\Desktop\EE200 Project\pdf_content.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        f.write(f"=== PAGE {i+1} ===\n")
        text = page.extract_text()
        f.write(text)
        f.write("\n\n")

print(f"Written to {output_path}")
print(f"Total pages: {len(reader.pages)}")
