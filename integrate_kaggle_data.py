import kagglehub
import pandas as pd
import os

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("prasad22/healthcare-dataset")
csv_path = os.path.join(path, "healthcare_dataset.csv")

print(f"Loading data from {csv_path}...")
df_kaggle = pd.read_csv(csv_path)

print("Transforming dataset to match our dashboard expectations...")
# Expected columns:
# Patient ID, Age, Gender, Disease, Admission Date, Discharge Date, Hospital Department, Treatment Cost

# Mapping Kaggle columns to our schema
df = pd.DataFrame()

# Generate a Patient ID sequence
df['Patient ID'] = [f"P{str(i).zfill(5)}" for i in range(1, len(df_kaggle) + 1)]
df['Age'] = df_kaggle['Age']
df['Gender'] = df_kaggle['Gender']
df['Disease'] = df_kaggle['Medical Condition']

# Parse dates
df['Admission Date'] = pd.to_datetime(df_kaggle['Date of Admission']).dt.strftime('%Y-%m-%d')
df['Discharge Date'] = pd.to_datetime(df_kaggle['Discharge Date']).dt.strftime('%Y-%m-%d')

# Map 'Admission Type' or 'Hospital' to Department (or create a mapping from Disease)
def condition_to_dept(condition):
    mapping = {
        'Diabetes': 'Endocrinology',
        'Asthma': 'Pulmonology',
        'Obesity': 'General Medicine',
        'Arthritis': 'Rheumatology',
        'Hypertension': 'Cardiology',
        'Cancer': 'Oncology'
    }
    return mapping.get(condition, 'General Medicine')

df['Hospital Department'] = df_kaggle['Medical Condition'].apply(condition_to_dept)

# Treatment Cost
df['Treatment Cost'] = df_kaggle['Billing Amount'].round(2)

# Save to our data directory, overwriting the synthetic data
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'hospital_patient_data.csv')

df.to_csv(output_path, index=False)
print(f"\nSuccess! Transformed {len(df)} records from Kaggle.")
print(f"Saved real dataset to {output_path}")

print("\nFirst 3 rows of the new dataset:")
print(df.head(3))
