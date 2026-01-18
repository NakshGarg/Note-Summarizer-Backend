from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import fitz
import os
import re

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Scholar's Lens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = genai.GenerativeModel("gemini-1.5-flash")

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class SummarizeRequest(BaseModel):
    text: str
    difficulty: str


class SummarizeResponse(BaseModel):
    short_notes: str
    bullet_points: str
    exam_summary: str


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception:
        raise HTTPException(status_code=400, detail="PDF text extraction failed")


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.split("\n")]
    lines = [
        line for line in lines
        if line
        and not re.fullmatch(r"\d+", line)
        and not re.match(r"^(Page|Chapter|Section)\s*\d+", line, re.I)
    ]
    return re.sub(r"\s+", " ", " ".join(lines))


def generate_notes(text: str, difficulty: str) -> dict:
    prompt = f"""
You are an academic notes generator.

Input Text:
\"\"\"{text}\"\"\"

Difficulty: {difficulty}

Rules:
- No emojis
- No extra information
- Use **bold** for keywords
- Exam-oriented
- Simple language

Output EXACTLY in this format:

### Short Notes
...

### Bullet Points
...

### Exam-Oriented Summary
...
"""

    response = model.generate_content(prompt)
    output = response.text or ""

    sections = {"short_notes": "", "bullet_points": "", "exam_summary": ""}

    parts = output.split("###")
    for part in parts:
        part = part.strip()
        if part.startswith("Short Notes"):
            sections["short_notes"] = part.replace("Short Notes", "").strip()
        elif part.startswith("Bullet Points"):
            sections["bullet_points"] = part.replace("Bullet Points", "").strip()
        elif part.startswith("Exam-Oriented Summary"):
            sections["exam_summary"] = part.replace("Exam-Oriented Summary", "").strip()

    return sections


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "Scholar's Lens API running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif file.filename.endswith(".txt"):
        text = file_bytes.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Only PDF and TXT supported")

    cleaned = clean_text(text)

    if not cleaned:
        raise HTTPException(status_code=400, detail="No valid text found")

    return {"text": cleaned}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    notes = generate_notes(req.text, req.difficulty)
    return SummarizeResponse(**notes)
