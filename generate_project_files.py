import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import os

# 1. Generate CSV
np.random.seed(42)
random.seed(42)

n_rows = 500
patient_ids = [f"P{str(i).zfill(4)}" for i in range(1, n_rows+1)]
ages = np.random.randint(1, 90, size=n_rows)
gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52], size=n_rows)
diseases = ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'COVID-19', 'Fracture', 'Pneumonia', 'None']
disease_col = np.random.choice(diseases, size=n_rows)

departments = ['Cardiology', 'Neurology', 'Orthopedics', 'General Medicine', 'Pediatrics', 'Emergency']
dept_col = []
for d in disease_col:
    if d == 'Heart Disease': dept_col.append('Cardiology')
    elif d == 'Fracture': dept_col.append('Orthopedics')
    elif d == 'Asthma' or d == 'Pneumonia' or d == 'COVID-19': dept_col.append('General Medicine')
    elif d == 'None': dept_col.append('Emergency')
    else: dept_col.append(np.random.choice(departments))

start_date = datetime(2023, 1, 1)
admission_dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(n_rows)]
stays = np.random.randint(1, 15, size=n_rows)
discharge_dates = [ad + timedelta(days=int(s)) for ad, s in zip(admission_dates, stays)]

costs = np.random.uniform(500, 15000, size=n_rows).round(2)

df = pd.DataFrame({
    'Patient ID': patient_ids,
    'Age': ages,
    'Gender': gender,
    'Disease': disease_col,
    'Admission Date': admission_dates,
    'Discharge Date': discharge_dates,
    'Hospital Department': dept_col,
    'Treatment Cost': costs
})

# Add some missing values and duplicates to test data cleaning
df.loc[10:20, 'Age'] = np.nan
df.loc[30:40, 'Disease'] = np.nan
df = pd.concat([df, df.iloc[0:5]], ignore_index=True)

os.makedirs('data', exist_ok=True)
df.to_csv('data/hospital_patient_data.csv', index=False)
print("Generated data/hospital_patient_data.csv")

# 2. Generate Jupyter Notebook JSON
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hospital Patient Data Analytics - EDA\n",
    "This notebook performs Exploratory Data Analysis on the hospital patient dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import sys\n",
    "import os\n",
    "sys.path.append(os.path.abspath('../src'))\n",
    "from data_cleaning import clean_data\n",
    "from visualization import plot_age_distribution, plot_disease_frequency\n",
    "\n",
    "# Load and clean data\n",
    "df = pd.read_csv('../data/hospital_patient_data.csv')\n",
    "df_clean = clean_data(df)\n",
    "df_clean.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# EDA: Age Distribution\n",
    "fig = plot_age_distribution(df_clean)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# EDA: Disease frequency\n",
    "fig = plot_disease_frequency(df_clean)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/eda_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Generated notebooks/eda_analysis.ipynb")
