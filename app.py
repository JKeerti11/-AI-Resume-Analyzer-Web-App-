import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber

# Load environment variables
load_dotenv()

# Configure Google Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if text.strip():
            return text.strip()

    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    # Fallback to OCR for image-based PDFs
    print("Falling back to OCR for image-based PDF.")
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed: {e}")

    return text.strip()


# Function to analyze resume using Gemini
def analyze_resume(resume_text, job_description=None):
    if not resume_text:
        return {"error": "Resume text is required for analysis."}

    # ✅ Correct, working Gemini model
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    base_prompt = f"""
    You are an experienced HR professional with deep technical knowledge.

    Evaluate the following resume in detail:
    - Does the candidate fit common job roles such as Data Scientist, Data Analyst,
      Software Developer, Full Stack Developer, AI Engineer, ML Engineer, DevOps, etc.?
    - List skills they already have
    - List missing skills they should improve
    - Recommend relevant online courses
    - Highlight strengths and weaknesses
    - Give an overall assessment

    Resume:
    {resume_text}
    """

    if job_description:
        base_prompt += f"""

        Now compare this resume with the job description below.
        Highlight:
        - Strengths
        - Weaknesses
        - Missing skills
        - Match percentage

        Job Description:
        {job_description}
        """

    response = model.generate_content(base_prompt)

    return response.text.strip()


# -------------------- STREAMLIT APP UI --------------------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and analyze it using Google Gemini AI")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

with col2:
    job_description = st.text_area("Paste Job Description (Optional)")


if uploaded_file is not None:
    st.success("✔ Resume uploaded successfully!")
else:
    st.warning("⚠ Please upload a PDF resume.")


st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)

if uploaded_file:
    with open("uploaded_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    resume_text = extract_text_from_pdf("uploaded_resume.pdf")

    if st.button("🔍 Analyze Resume"):
        with st.spinner("Analyzing with Gemini..."):
            try:
                analysis = analyze_resume(resume_text, job_description)
                st.success("✅ Analysis Completed!")
                st.write(analysis)

            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")


# Footer
st.markdown("---")
st.markdown(
    """<p style='text-align: center;'>Powered by <b>Google Gemini AI</b> & <b>Streamlit</b></p>""",
    unsafe_allow_html=True
)
