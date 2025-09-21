import streamlit as st
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer, util
import re

# Load sentence-transformers model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------- Utils -------------------

def extract_text_from_pdf(file):
    """Extract text from PDF using PyMuPDF"""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX using python-docx"""
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    """Lowercase & remove special chars"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text

def keyword_score(resume_text, jd_text):
    """Hard keyword matching"""
    resume_words = set(clean_text(resume_text).split())
    jd_words = set(clean_text(jd_text).split())

    common = resume_words.intersection(jd_words)
    return round((len(common) / len(jd_words)) * 100, 2)

def semantic_score(resume_text, jd_text):
    """Soft semantic similarity with embeddings"""
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(resume_emb, jd_emb).item()
    return round(score * 100, 2)

def final_verdict(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Resume Relevance Checker", page_icon="📄")
st.title("📄 Automated Resume Relevance Check System")

jd_text = st.text_area("Paste Job Description here")

uploaded_resume = st.file_uploader("Upload your Resume", type=["pdf", "docx"])

if uploaded_resume and jd_text.strip():
    if uploaded_resume.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_resume)
    else:
        resume_text = extract_text_from_docx(uploaded_resume)

    st.subheader("Analysis Results")

    # Step 1: Hard match
    hard = keyword_score(resume_text, jd_text)

    # Step 2: Soft match
    soft = semantic_score(resume_text, jd_text)

    # Weighted score (40% keywords, 60% semantic)
    final_score = round((0.4 * hard) + (0.6 * soft), 2)
    verdict = final_verdict(final_score)

    st.metric("Relevance Score", f"{final_score}/100")
    st.write(f"**Verdict:** {verdict} suitability")

    st.write("### Breakdown")
    st.write(f"- Hard Keyword Match: {hard}/100")
    st.write(f"- Semantic Match: {soft}/100")

    # Suggestions (very simple for now)
    st.write("### Suggestions")
    if final_score < 50:
        st.warning("⚠️ Add more relevant skills & keywords from the job description.")
    elif final_score < 75:
        st.info("👍 Good! Try adding projects/certifications related to JD.")
    else:
        st.success("✅ Strong fit! Resume matches the job description well.")
