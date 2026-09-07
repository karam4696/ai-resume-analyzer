import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import re


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📝",
    layout="wide"
)


# =========================
# API KEY
# =========================

api_key = st.secrets["GEMINI_API_KEY"]


# =========================
# SESSION STATE
# =========================

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume = ""

if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""


# =========================
# TITLE
# =========================

st.title("AI Resume Analyzer 📝")
st.caption("Analyze your resume against a job description using AI + ATS similarity.")


# =========================
# LOAD ATS MODEL
# =========================

@st.cache_resource
def load_ats_model():
    return SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
    )


# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_pdf_text(uploaded_file):

    try:
        extracted_text = extract_text(uploaded_file)

        if not extracted_text.strip():
            st.error(
                "PDF se text extract nahi ho paya. "
                "Please check that your PDF contains selectable text."
            )
            return ""

        return extracted_text

    except Exception as e:

        st.error("Error extracting text from PDF.")
        st.exception(e)

        return ""


# =========================
# ATS SIMILARITY
# =========================

def calculate_similarity_bert(text1, text2):

    try:

        ats_model = load_ats_model()

        embeddings1 = ats_model.encode([text1])
        embeddings2 = ats_model.encode([text2])

        similarity = cosine_similarity(
            embeddings1,
            embeddings2
        )[0][0]

        return float(similarity)

    except Exception as e:

        st.error("ATS similarity calculation failed.")
        st.exception(e)

        return 0.0


# =========================
# GEMINI AI REPORT
# =========================

def get_report(resume, job_desc):

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    prompt = f"""
You are an expert AI Resume Analyzer and ATS consultant.

Analyze the candidate's resume against the provided job description.

Consider:

- Required skills
- Technical skills
- Education
- Experience
- Projects
- Certifications
- Responsibilities
- Tools and technologies
- Relevant keywords
- Other important requirements

For every important requirement:

1. Give a score out of 5.
2. Put the score at the beginning.
3. Use:
   ✅ = Resume aligns with the requirement
   ❌ = Resume does not align with the requirement
   ⚠️ = Cannot clearly determine from the resume
4. Explain the reason clearly.

At the end provide:

Suggestions to improve your resume:

Give practical suggestions that can improve the candidate's chances
of matching this job description.

IMPORTANT:
Do not invent experience, skills, education or certifications that are
not present in the resume.

OUTPUT FORMAT:

3/5 ✅ Technical Skills

Explanation...

4/5 ⚠️ Experience

Explanation...

2/5 ❌ Education

Explanation...

Suggestions to improve your resume:

- Suggestion 1
- Suggestion 2
- Suggestion 3


========================
CANDIDATE RESUME
========================

{resume}


========================
JOB DESCRIPTION
========================

{job_desc}
"""

    try:

        response = client.chat.completions.create(
            model="gemini-3.6-flash",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        if response.choices:
            return response.choices[0].message.content

        return ""

    except Exception as e:

        st.error("Gemini API Error")
        st.exception(e)

        return ""


# =========================
# EXTRACT SCORES
# =========================

def extract_scores(text):

    if not text:
        return []

    pattern = r"(\d+(?:\.\d+)?)/5"

    matches = re.findall(pattern, text)

    scores = [
        float(match)
        for match in matches
        if 0 <= float(match) <= 5
    ]

    return scores


# =========================
# MAIN FORM
# =========================

if not st.session_state.form_submitted:

    st.subheader("Upload Your Resume")

    with st.form("resume_form"):

        resume_file = st.file_uploader(
            "Upload Resume / CV",
            type=["pdf"]
        )

        job_desc = st.text_area(
            "Enter Job Description",
            placeholder="Paste the complete job description here...",
            height=250
        )

        submitted = st.form_submit_button(
            "🚀 Analyze Resume",
            use_container_width=True
        )

        if submitted:

            if not resume_file or not job_desc.strip():

                st.warning(
                    "Please upload your resume and enter the job description."
                )

            else:

                with st.spinner("Extracting resume information..."):

                    resume_text = extract_pdf_text(resume_file)

                if resume_text:

                    st.session_state.resume = resume_text
                    st.session_state.job_desc = job_desc
                    st.session_state.form_submitted = True

                    st.rerun()


# =========================
# ANALYSIS PAGE
# =========================

if st.session_state.form_submitted:

    st.subheader("📊 Resume Analysis")

    # ATS SCORE
    with st.spinner("Calculating ATS similarity..."):

        ats_score = calculate_similarity_bert(
            st.session_state.resume,
            st.session_state.job_desc
        )

    # AI REPORT
    with st.spinner("Generating AI analysis..."):

        report = get_report(
            st.session_state.resume,
            st.session_state.job_desc
        )

    # Extract scores
    report_scores = extract_scores(report)

    if report_scores:

        avg_score = sum(report_scores) / len(report_scores)

    else:

        avg_score = 0


    # =========================
    # SCORE CARDS
    # =========================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "ATS Similarity Score",
            f"{ats_score * 100:.1f}%"
        )

    with col2:

        st.metric(
            "AI Resume Score",
            f"{avg_score / 5 * 100:.1f}%"
        )


    st.divider()


    # =========================
    # AI REPORT
    # =========================

    st.subheader("🤖 AI Generated Analysis")

    if report:

        st.markdown(report)

    else:

        st.error(
            "AI report generate nahi hua. "
            "Please check your Gemini API key and model configuration."
        )


    # =========================
    # DOWNLOAD
    # =========================

    if report:

        st.download_button(
            label="⬇️ Download Report",
            data=report,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )


    # =========================
    # NEW ANALYSIS
    # =========================

    if st.button(
        "🔄 Analyze Another Resume",
        use_container_width=True
    ):

        st.session_state.form_submitted = False
        st.session_state.resume = ""
        st.session_state.job_desc = ""

        st.rerun()