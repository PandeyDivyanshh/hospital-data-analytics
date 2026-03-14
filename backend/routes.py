from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv
import pandas as pd
import os
import sys

# Load environment variables
load_dotenv()

# Ensure src is in the path to import data_cleaning
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_cleaning import clean_data

router = APIRouter()

# ── Groq Client ──────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Pydantic Models ──────────────────────────────────────────────────────────
class PatientRecord(BaseModel):
    patient_name: str
    age: int
    gender: str
    disease: str
    hospital_department: str
    admission_date: str      # YYYY-MM-DD
    discharge_date: str      # YYYY-MM-DD
    treatment_cost: float

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

# ── Data Helpers ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'hospital_patient_data.csv')

def load_and_clean_data() -> pd.DataFrame:
    """Loads the CSV and cleans it using the existing src/data_cleaning module."""
    try:
        df = pd.read_csv(DATA_PATH)
        return clean_data(df)
    except FileNotFoundError:
        return pd.DataFrame()

def get_data_summary(df: pd.DataFrame) -> str:
    """Generates a concise text summary of the dataset for AI context."""
    if df.empty:
        return "No patient data available."
    summary = (
        f"Hospital dataset has {len(df)} patients. "
        f"Age range: {int(df['Age'].min())}-{int(df['Age'].max())} (avg {df['Age'].mean():.1f}). "
        f"Departments: {', '.join(df['Hospital Department'].unique()[:6])}. "
        f"Top diseases: {', '.join(df['Disease'].value_counts().head(5).index.tolist())}. "
        f"Avg treatment cost: ${df['Treatment Cost'].mean():,.0f}. "
        f"Avg hospital stay: {df['Hospital Stay Days'].mean():.1f} days."
    )
    return summary

# ── Health ───────────────────────────────────────────────────────────────────
@router.get("/health")
def health_check():
    return {"status": "ok"}

# ── GET Endpoints ────────────────────────────────────────────────────────────
@router.get("/api/data")
def get_all_data():
    df = load_and_clean_data()
    if df.empty:
        return {"error": "Data file not found"}
    for col in ['Admission Date', 'Discharge Date']:
        if col in df.columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
    return df.to_dict(orient="records")

@router.get("/api/data/filter")
def get_filtered_data(
    age_min: int = Query(0),
    age_max: int = Query(120),
    departments: Optional[List[str]] = Query(None)
):
    df = load_and_clean_data()
    if df.empty:
        return {"error": "Data file not found"}
    filtered_df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]
    if departments:
        filtered_df = filtered_df[filtered_df['Hospital Department'].isin(departments)]
    for col in ['Admission Date', 'Discharge Date']:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].dt.strftime('%Y-%m-%d')
    return filtered_df.to_dict(orient="records")

@router.get("/api/departments")
def get_departments():
    df = load_and_clean_data()
    if df.empty:
        return []
    return df['Hospital Department'].unique().tolist()

@router.get("/api/diseases")
def get_diseases():
    df = load_and_clean_data()
    if df.empty:
        return []
    return df['Disease'].unique().tolist()

# ── POST: Add Patient ───────────────────────────────────────────────────────
@router.post("/api/data")
def add_patient(record: PatientRecord):
    """Appends a new patient record to the CSV file."""
    try:
        # Build a new row matching the CSV columns
        new_row = {
            "Patient ID": f"PID-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            "Patient Name": record.patient_name,
            "Age": record.age,
            "Gender": record.gender,
            "Disease": record.disease,
            "Hospital Department": record.hospital_department,
            "Admission Date": record.admission_date,
            "Discharge Date": record.discharge_date,
            "Treatment Cost": record.treatment_cost
        }
        
        new_df = pd.DataFrame([new_row])
        
        if os.path.exists(DATA_PATH):
            new_df.to_csv(DATA_PATH, mode='a', header=False, index=False)
        else:
            new_df.to_csv(DATA_PATH, index=False)
        
        return {"status": "success", "message": f"Patient '{record.patient_name}' added successfully.", "patient_id": new_row["Patient ID"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

# ── POST: AI Chat ───────────────────────────────────────────────────────────
@router.post("/api/chat")
def ai_chat(req: ChatRequest):
    """Sends a user query with dataset context to the Groq LLM."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    # Get dataset context
    df = load_and_clean_data()
    data_context = get_data_summary(df)
    
    system_prompt = (
        "You are a helpful Hospital Data Analytics AI Assistant. "
        "You help users understand patient data, trends, and medical insights. "
        "Be concise, professional, and cite data when possible.\n\n"
        f"Current Dataset Summary:\n{data_context}"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    for msg in (req.history or []):
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    
    # Add the latest user message
    messages.append({"role": "user", "content": req.message})
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )
        ai_reply = response.choices[0].message.content
        return {"reply": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
