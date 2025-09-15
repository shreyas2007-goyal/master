import streamlit as st
import pandas as pd
import os
import tempfile
import re
import random
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
from PIL import Image
import io

# Optional: OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

# Optional: OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        OPENAI_AVAILABLE = False
except Exception:
    openai = None
    OPENAI_AVAILABLE = False


st.set_page_config(page_title="Exam Evaluation Transparency", layout="wide")
st.title("Exam Evaluation Transparency")
st.write("This app evaluates exam answers either manually, from scanned sheets, or by capturing with live camera.")


# =========================
# Mock evaluation
# =========================
def mock_evaluation(answer):
    feedback_samples = [
        ("Strong explanation but missed key point.", 7),
        ("Excellent coverage with examples.", 9),
        ("Weak structure, needs clarity.", 5),
        ("Good attempt but lacking depth.", 6),
        ("Outstanding answer with clear reasoning.", 10)
    ]
    feedback, score = random.choice(feedback_samples)
    return score, feedback


# =========================
# OCR
# =========================
def extract_text_from_image_bytes(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return None, f"Could not open image: {e}"

    if TESSERACT_AVAILABLE:
        try:
            text = pytesseract.image_to_string(image)
            return text.strip(), None
        except Exception as e:
            return None, f"pytesseract error: {e}"
    else:
        return None, "OCR not available"


# =========================
# OpenAI evaluation
# =========================
def call_openai_evaluate(extracted_text):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI API not configured.")

    prompt = f"""
    You are an exam evaluator. The student's answer is below:

    {extracted_text}

    Provide feedback in this format:
    Marks: X/10
    Strengths:
    - point
    Weaknesses:
    - point
    Improvements:
    - point
    """

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1
    )
    return resp["choices"][0]["message"]["content"]


# =========================
# Section extraction
# =========================
def extract_section_items(feedback, header):
    regex = rf"{header}\s*:\s*(.*?)(?=\n[A-Z][a-zA-Z ]+?:|\Z)"
    m = re.search(regex, feedback, flags=re.IGNORECASE | re.DOTALL)
    items = []
    if m:
        block = m.group(1).strip()
        parts = re.split(r"\n[\-\•]?\s*", block)
        for p in parts:
            p = p.strip().lstrip("-• ").strip()
            if p:
                items.append(p)
    return items


# =========================
# Download helpers
# =========================
def download_report_csv(df, student_name):
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output


def download_report_pdf(feedback):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(80, 760, "Exam Evaluation Report")
    y = 740
    lines = feedback.splitlines()
    for line in lines:
        if y < 60:
            c.showPage()
            y = 740
        c.drawString(80, y, line.strip())
        y -= 16
    c.save()
    with open(tmp_file.name, "rb") as f:
        return f.read()


# =========================
# Sidebar
# =========================
st.sidebar.header("Settings")
student_name = st.sidebar.text_input("Student Name", "John Doe")
input_mode = st.sidebar.radio("Input Mode", ["Manual Answers", "Upload Scanned Sheet", "Live Camera Capture"])
mock_mode = st.sidebar.checkbox("Enable Mock Mode (No API Key Required)", value=True)
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)


# =========================
# Manual Mode
# =========================
if input_mode == "Manual Answers":
    num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=3, step=1)

    answers = {}
    for i in range(1, num_questions + 1):
        answers[f"Q{i}"] = st.text_area(f"Answer {i}", "")

    if st.button("Evaluate"):
        results = []
        for q, ans in answers.items():
            if ans.strip():
                if not mock_mode and OPENAI_AVAILABLE:
                    feedback = call_openai_evaluate(ans)
                    score_match = re.search(r"Marks\s*:\s*(\d+)", feedback)
                    score = int(score_match.group(1)) if score_match else 0
                else:
                    score, feedback = mock_evaluation(ans)

                results.append({"Question": q, "Answer": ans, "Score": score, "Feedback": feedback})

        if results:
            df = pd.DataFrame(results)
            st.subheader("Evaluation Results")
            st.dataframe(df, use_container_width=True)

            csv = download_report_csv(df, student_name)
            st.download_button("Download CSV Report", data=csv, file_name=f"{student_name}_evaluation.csv", mime="text/csv")


# =========================
# Upload Mode
# =========================
if input_mode == "Upload Scanned Sheet":
    uploaded_file = st.file_uploader("Upload Answer Sheet (PNG/JPG)", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file and st.button("Analyze Uploaded"):
        file_bytes = uploaded_file.read()
        extracted_text, ocr_message = None, None
        if uploaded_file.type.startswith("image/"):
            extracted_text, ocr_message = extract_text_from_image_bytes(file_bytes)

        # Evaluation
        feedback = None
        if extracted_text and not mock_mode and OPENAI_AVAILABLE:
            feedback = call_openai_evaluate(extracted_text)

        if not feedback:
            feedback = (
                "Marks: 7/10\n\n"
                "Strengths:\n- Good explanation.\n- Correct steps.\n\n"
                "Weaknesses:\n- Missed final step.\n- Minor errors.\n\n"
                "Improvements:\n- Add example.\n- Present final answer clearly."
            )

        st.code(feedback, language="text")
        pdf_bytes = download_report_pdf(feedback)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="evaluation_report.pdf", mime="application/pdf")


# =========================
# Live Camera Mode
# =========================
if input_mode == "Live Camera Capture":
    picture = st.camera_input("Take a picture of the answer sheet")

    if picture and st.button("Analyze Photo"):
        img_bytes = picture.getvalue()
        extracted_text, ocr_message = extract_text_from_image_bytes(img_bytes)

        # Evaluation
        feedback = None
        if extracted_text and not mock_mode and OPENAI_AVAILABLE:
            feedback = call_openai_evaluate(extracted_text)

        if not feedback:
            feedback = (
                "Marks: 6/10\n\n"
                "Strengths:\n- Clear handwriting.\n- Key concepts mentioned.\n\n"
                "Weaknesses:\n- Incomplete explanation.\n- Lacks examples.\n\n"
                "Improvements:\n- Add diagrams.\n- Write detailed steps."
            )

        st.code(feedback, language="text")
        pdf_bytes = download_report_pdf(feedback)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="evaluation_report.pdf", mime="application/pdf")
