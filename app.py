import gradio as gr
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model (will auto-download)
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(file):
    """Extract text from PDF or DOCX"""
    text = ""
    if file.name.endswith(".pdf"):
        doc = fitz.open(file.name)
        for page in doc:
            text += page.get_text()
    elif file.name.endswith(".docx"):
        doc = Document(file.name)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def check_resume(resume_file, jd_text):
    resume_text = extract_text(resume_file)
    # Simple semantic similarity using embeddings
    embeddings_resume = model.encode(resume_text, convert_to_tensor=True)
    embeddings_jd = model.encode(jd_text, convert_to_tensor=True)
    score = util.cos_sim(embeddings_resume, embeddings_jd).item() * 100
    verdict = "High suitability" if score > 70 else "Medium suitability" if score > 40 else "Low suitability"
    return f"Relevance Score: {score:.2f}/100\nVerdict: {verdict}"

# Gradio interface
iface = gr.Interface(
    fn=check_resume,
    inputs=[gr.File(label="Upload Resume (PDF/DOCX)"), gr.Textbox(label="Paste Job Description")],
    outputs="text",
    title="Automated Resume Relevance Checker",
    description="Upload your resume and paste the job description to get a relevance score and verdict."
)

iface.launch()
