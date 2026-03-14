"""
Streamlit Dashboard for Hospital Patient Data Analytics
(FastAPI + AI Integration Version)
"""
import streamlit as st
import pandas as pd
import requests
import sys
import os
from datetime import date, timedelta

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import visualization as viz

# Set page config
st.set_page_config(page_title="Hospital Analytics Dashboard", page_icon="🏥", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"

# ── Helper Functions ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_departments():
    try:
        res = requests.get(f"{API_BASE_URL}/api/departments")
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return []

@st.cache_data(ttl=30)
def fetch_diseases():
    try:
        res = requests.get(f"{API_BASE_URL}/api/diseases")
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return []

def fetch_filtered_data(age_min, age_max, departments):
    try:
        params = {"age_min": age_min, "age_max": age_max}
        if departments:
            params["departments"] = departments
        res = requests.get(f"{API_BASE_URL}/api/data/filter", params=params)
        res.raise_for_status()
        data = res.json()
        if isinstance(data, dict) and "error" in data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if not df.empty:
            if 'Admission Date' in df.columns:
                df['Admission Date'] = pd.to_datetime(df['Admission Date'])
            if 'Discharge Date' in df.columns:
                df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
        return df
    except requests.exceptions.RequestException:
        return pd.DataFrame()

# ── Check Backend ────────────────────────────────────────────────────────────
all_departments = fetch_departments()
if not all_departments:
    st.error("⚠️ Cannot connect to the FastAPI backend at http://127.0.0.1:8000. Please start the backend server first.")
    st.stop()

all_diseases = fetch_diseases()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏥 Hospital Patient Data Analytics Dashboard")
st.caption("Interactive dashboard powered by FastAPI backend & Groq AI")

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filter Data")
age_range = st.sidebar.slider("Age Range", 0, 120, (0, 120))
selected_departments = st.sidebar.multiselect("Departments", options=all_departments, default=all_departments)

filtered_df = fetch_filtered_data(age_range[0], age_range[1], selected_departments)

if filtered_df.empty:
    st.warning("No data found for these filters.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.success(f"📊 Showing **{len(filtered_df)}** patients")

# ── Metrics ──────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Patients", f"{len(filtered_df):,}")
m2.metric("Average Age", f"{filtered_df['Age'].mean():.1f} yrs")
m3.metric("Avg Stay", f"{filtered_df['Hospital Stay Days'].mean():.1f} days")
m4.metric("Avg Cost", f"${filtered_df['Treatment Cost'].mean():,.0f}")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_demo, tab_trends, tab_cost, tab_entry, tab_ai = st.tabs([
    "📊 Demographics & Diseases",
    "📈 Admission Trends",
    "💰 Cost Analytics",
    "📝 Add Patient",
    "🤖 AI Assistant"
])

# ── Tab 1: Demographics ─────────────────────────────────────────────────────
with tab_demo:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Patient Age Distribution")
        st.pyplot(viz.plot_age_distribution(filtered_df))
    with col_b:
        st.subheader("Department-wise Distribution")
        st.pyplot(viz.plot_department_distribution(filtered_df))
    st.subheader("Top Diseases Frequency")
    st.pyplot(viz.plot_disease_frequency(filtered_df))

# ── Tab 2: Trends ────────────────────────────────────────────────────────────
with tab_trends:
    st.subheader("Monthly Admission Trends")
    st.plotly_chart(viz.plot_admission_trends_plotly(filtered_df), use_container_width=True)

# ── Tab 3: Cost ──────────────────────────────────────────────────────────────
with tab_cost:
    st.subheader("Average Treatment Cost by Department & Disease")
    st.pyplot(viz.plot_cost_heatmap(filtered_df))

# ── Tab 4: Data Entry Form ───────────────────────────────────────────────────
with tab_entry:
    st.subheader("📝 Add a New Patient Record")
    st.markdown("Fill in the form below and click **Submit** to add a new patient to the database.")
    
    with st.form("patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            disease = st.selectbox("Disease", options=all_diseases if all_diseases else ["Flu", "Diabetes", "Heart Disease", "Cancer", "Asthma"])
        with col2:
            department = st.selectbox("Hospital Department", options=all_departments)
            admission_date = st.date_input("Admission Date", value=date.today())
            discharge_date = st.date_input("Discharge Date", value=date.today() + timedelta(days=5))
            treatment_cost = st.number_input("Treatment Cost ($)", min_value=0.0, value=5000.0, step=100.0)
        
        submitted = st.form_submit_button("🚀 Submit Patient Record", use_container_width=True)
        
        if submitted:
            if not patient_name.strip():
                st.error("Patient name is required!")
            elif discharge_date < admission_date:
                st.error("Discharge date cannot be before admission date!")
            else:
                payload = {
                    "patient_name": patient_name,
                    "age": age,
                    "gender": gender,
                    "disease": disease,
                    "hospital_department": department,
                    "admission_date": str(admission_date),
                    "discharge_date": str(discharge_date),
                    "treatment_cost": treatment_cost
                }
                try:
                    res = requests.post(f"{API_BASE_URL}/api/data", json=payload)
                    res.raise_for_status()
                    result = res.json()
                    st.success(f"✅ {result['message']} (ID: {result['patient_id']})")
                    st.cache_data.clear()
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to add patient: {e}")

# ── Tab 5: AI Assistant ──────────────────────────────────────────────────────
with tab_ai:
    st.subheader("🤖 AI Health Data Assistant")
    st.markdown("Ask questions about the hospital data — powered by **Groq Llama 3.1**.")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display existing messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about patient trends, costs, diseases...")
    
    if user_input:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Call backend AI endpoint
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "message": user_input,
                        "history": st.session_state.chat_history[:-1]  # send previous context
                    }
                    res = requests.post(f"{API_BASE_URL}/api/chat", json=payload)
                    res.raise_for_status()
                    ai_reply = res.json()["reply"]
                    st.markdown(ai_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                except requests.exceptions.RequestException as e:
                    st.error(f"AI Error: {e}")

st.markdown("---")
st.markdown("### 📋 Raw Data Preview")
st.dataframe(filtered_df.head(10), use_container_width=True)
