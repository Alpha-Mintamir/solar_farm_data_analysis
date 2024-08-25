# Solar Farm Data Analysis

## Overview
This project analyzes solar farm data from Benin, Sierra Leone, and Togo to uncover key insights about solar irradiance and related parameters. The analysis involves data preprocessing, statistical analysis, and various visualizations to help understand the patterns and trends in the solar data. The results are presented through an interactive Streamlit dashboard(https://alpha-mintamir-solar-farm-data-analysis-appmain-y6rsre.streamlit.app/) , making it easy to explore the data and findings.

## Project Structure
The repository is organized as follows:

- **data/**: Contains the CSV files with the solar farm data:
  - `benin-malanville.csv`
  - `sierraleone-bumbuna.csv`
  - `togo-dapaong_qc.csv`
  
- **app.py**: The main script for the Streamlit application. This script loads the data, performs analysis, and renders the interactive dashboard.

- **requirements.txt**: Lists the Python dependencies required to run the project, including libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, and `streamlit`.

- **README.md**: Provides an overview of the project, setup instructions, and details about the Streamlit dashboard.

## Streamlit Dashboard
Streamlit is an open-source app framework for creating data-driven web applications in Python. In this project, we use Streamlit to create an interactive dashboard that allows users to explore the solar farm data. The dashboard includes several features:

- **Data Previews**: Display the first few rows of the datasets for Benin, Sierra Leone, and Togo.
- **Summary Statistics**: Show descriptive statistics for the datasets, including measures of central tendency and variability.
- **Missing Values**: Identify and display the number of missing values in each dataset.
- **Outlier Detection**: Detect and display outliers in the Global Horizontal Irradiance (GHI) values using the Z-score method.
- **Visualizations**: 
  - **Time Series Plots**: Visualize the temporal patterns of GHI.
  - **Correlation Heatmaps**: Explore the relationships between solar parameters like GHI, DNI, and DHI.
  - **Wind Polar Plots**: Display the distribution of wind speed and direction.
  - **Histograms**: Analyze the distribution of GHI values.

The Streamlit dashboard is designed to be user-friendly, allowing users to interact with the data and gain insights without needing to write code.

## Setup Instructions
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/solar-farm-data-analysis.git
   cd solar-farm-data-analysis
## Install Dependencies

Create a virtual environment (optional but recommended), and install the required packages:


python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
pip install -r requirements.txt

## Run the Streamlit App 

Launch the Streamlit dashboard:

```bash
streamlit run app.py
## Access the Dashboard

After running the command, Streamlit will start a local server. You can access the dashboard in your web browser at:http://localhost:8501

