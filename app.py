# app.py
import fitz  # PyMuPDF
import docx2txt
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# Load model once
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_resume_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(file.name)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file.name)
    else:
        return "Unsupported file type"

def analyze_resume(resume_file, jd_text):
    resume_text = extract_resume_text(resume_file)
    
    # Hard match (simple keyword count)
    jd_keywords = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    hard_score = len(jd_keywords & resume_words)/max(len(jd_keywords),1) * 100
    
    # Semantic match
    embeddings_jd = model.encode(jd_text, convert_to_tensor=True)
    embeddings_resume = model.encode(resume_text, convert_to_tensor=True)
    sem_score = util.cos_sim(embeddings_jd, embeddings_resume).item() * 100
    
    final_score = (0.5*hard_score + 0.5*sem_score)
    
    # Verdict
    if final_score > 80:
        verdict = "High suitability"
    elif final_score > 50:
        verdict = "Medium suitability"
    else:
        verdict = "Low suitability"
    
    return f"Relevance Score: {final_score:.2f}/100\nVerdict: {verdict}"

# Gradio interface
iface = gr.Interface(
    fn=analyze_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF/DOCX)"),
        gr.Textbox(label="Paste Job Description Here", lines=10)
    ],
    outputs="text",
    title="Automated Resume Relevance Checker",
    description="Upload a resume and paste the job description to get relevance score and verdict."
)

iface.launch()
