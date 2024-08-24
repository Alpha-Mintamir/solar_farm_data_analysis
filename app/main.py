import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

base_dir = os.path.dirname(os.path.abspath(__file__))

# Adjust the path to the data files
data_paths = {
    'benin': os.path.join(base_dir, '..', 'data', 'benin-malanville.csv'),
    'sierraleone': os.path.join(base_dir, '..', 'data', 'sierraleone-bumbuna.csv'),
    'togo': os.path.join(base_dir, '..', 'data', 'togo-dapaong_qc.csv')
}

# Normalize paths (to handle '..' correctly)
data_paths = {key: os.path.normpath(path) for key, path in data_paths.items()}
# Print file sizes
file_sizes = {key: os.path.getsize(path) for key, path in data_paths.items()}
print(file_sizes)

def load_data():
    def optimize_memory_usage(df):
        # Optimize memory usage for numerical columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    
    # Load data and optimize memory usage
    df_benin = pd.read_csv(data_paths['benin'])
    df_sierraleone = pd.read_csv(data_paths['sierraleone'])
    df_togo = pd.read_csv(data_paths['togo'])
    
    df_benin = optimize_memory_usage(df_benin)
    df_sierraleone = optimize_memory_usage(df_sierraleone)
    df_togo = optimize_memory_usage(df_togo)
    
    return df_benin, df_sierraleone, df_togo

def detect_outliers(df, column):
    # Z-score method to detect outliers
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return df[(z_scores > 3)]

def plot_time_series(df, column):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df[column], label=column)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title(f'Time Series of {column}')
    plt.legend()
    st.pyplot(plt)

def plot_correlation_heatmap(df):
    corr = df[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

def plot_wind_polar(df):
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection='polar')
    theta = np.deg2rad(df['WD'].dropna())
    r = df['WS'].dropna()
    ax.scatter(theta, r, c='b', alpha=0.5)
    plt.title('Wind Speed Distribution')
    st.pyplot(plt)

def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    st.pyplot(plt)

# Load the data
df_benin, df_sierraleone, df_togo = load_data()

# Streamlit app
st.title('Solar Farm Data Analysis')

# Display data
st.subheader('Benin Data Preview')
st.write(df_benin.head())
st.subheader('Sierra Leone Data Preview')
st.write(df_sierraleone.head())
st.subheader('Togo Data Preview')
st.write(df_togo.head())

# Summary statistics
st.subheader('Benin Summary Statistics')
st.write(df_benin.describe())
st.subheader('Sierra Leone Summary Statistics')
st.write(df_sierraleone.describe())
st.subheader('Togo Summary Statistics')
st.write(df_togo.describe())

# Missing values
st.subheader('Benin Missing Values')
st.write(df_benin.isnull().sum())
st.subheader('Sierra Leone Missing Values')
st.write(df_sierraleone.isnull().sum())
st.subheader('Togo Missing Values')
st.write(df_togo.isnull().sum())

# Outlier detection
st.subheader('Benin Outliers')
st.write(detect_outliers(df_benin, 'GHI'))
st.subheader('Sierra Leone Outliers')
st.write(detect_outliers(df_sierraleone, 'GHI'))
st.subheader('Togo Outliers')
st.write(detect_outliers(df_togo, 'GHI'))

# Plot time series
st.subheader('Benin GHI Time Series')
plot_time_series(df_benin, 'GHI')
st.subheader('Sierra Leone GHI Time Series')
plot_time_series(df_sierraleone, 'GHI')
st.subheader('Togo GHI Time Series')
plot_time_series(df_togo, 'GHI')

# Plot correlation heatmaps
st.subheader('Benin Correlation Heatmap')
plot_correlation_heatmap(df_benin)
st.subheader('Sierra Leone Correlation Heatmap')
plot_correlation_heatmap(df_sierraleone)
st.subheader('Togo Correlation Heatmap')
plot_correlation_heatmap(df_togo)

# Plot wind polar plots
st.subheader('Benin Wind Polar Plot')
plot_wind_polar(df_benin)
st.subheader('Sierra Leone Wind Polar Plot')
plot_wind_polar(df_sierraleone)
st.subheader('Togo Wind Polar Plot')
plot_wind_polar(df_togo)

# Plot histograms
st.subheader('Benin GHI Histogram')
plot_histogram(df_benin, 'GHI')
st.subheader('Sierra Leone GHI Histogram')
plot_histogram(df_sierraleone, 'GHI')
st.subheader('Togo GHI Histogram')
plot_histogram(df_togo, 'GHI')
