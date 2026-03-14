"""
Data Visualization Module for Hospital Patient Data
Provides functions to create modular visualizations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_age_distribution(df: pd.DataFrame):
    """Plots the distribution of patient ages."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax, color='skyblue')
    ax.set_title('Patient Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    return fig

def plot_disease_frequency(df: pd.DataFrame):
    """Plots the frequency of top diseases."""
    fig, ax = plt.subplots(figsize=(10, 6))
    disease_counts = df['Disease'].value_counts().head(10)
    sns.barplot(x=disease_counts.values, y=disease_counts.index, ax=ax, palette='viridis')
    ax.set_title('Top Diseases Frequency')
    ax.set_xlabel('Number of Patients')
    ax.set_ylabel('Disease')
    return fig

def plot_department_distribution(df: pd.DataFrame):
    """Plots the distribution of patients across departments as a pie chart."""
    dept_counts = df['Hospital Department'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax.set_title('Department-wise Patient Distribution')
    return fig

def plot_admission_trends_plotly(df: pd.DataFrame):
    """Plots time-series admission trends using Plotly for interactivity."""
    # Count admissions per month
    df_trend = df.copy()
    df_trend['Month_Year'] = df_trend['Admission Date'].dt.to_period('M').astype(str)
    trend_counts = df_trend.groupby('Month_Year').size().reset_index(name='Admissions')
    trend_counts = trend_counts.sort_values('Month_Year')
    
    fig = px.line(trend_counts, x='Month_Year', y='Admissions', markers=True, 
                  title='Monthly Admission Trends', 
                  labels={'Month_Year': 'Month', 'Admissions': 'Number of Admissions'})
    return fig

def plot_cost_heatmap(df: pd.DataFrame):
    """Plots a heatmap for average treatment cost by department and disease."""
    pivot_df = df.pivot_table(values='Treatment Cost', index='Hospital Department', columns='Disease', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title('Average Treatment Cost by Department and Disease')
    return fig
