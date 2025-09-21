# 📄 Automated Resume Relevance Check System (GenAI Project)

🚀 Built for **Code4EdTech Hackathon @ Innomatics Research Labs 2025**  

This project helps recruiters and candidates by **automatically checking the relevance of resumes against job descriptions** using **Generative AI + NLP**.


---

## ✨ Features
- 📤 Upload Resume (PDF/DOCX supported)
- 📋 Paste Job Description
- 🔍 Extracts & cleans text using **PyMuPDF + python-docx**
- 🤖 Uses **Sentence Transformers (MiniLM)** for semantic similarity
- ✅ Calculates:
  - Hard Keyword Match %
  - Semantic Similarity %
  - Weighted Final Score (Verdict: High / Medium / Low)
- 💡 Provides Suggestions for Resume Improvement

---

## 🖥️ Tech Stack
- **Frontend/UI** → [Streamlit](https://streamlit.io/)  
- **NLP Model** → [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- **Libraries** → PyMuPDF, python-docx, scikit-learn, transformers, torch  
- **Language** → Python 3.10+  

---

## 📂 Project Structure
