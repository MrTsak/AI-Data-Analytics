# AI Data Analytics - Diabetes Prediction Dashboard

## Overview

This application is an interactive diabetes prediction dashboard that combines data analytics with machine learning to provide insights into diabetes risk factors and predictions. Built with Python and CustomTkinter, it offers a comprehensive suite of tools for exploring diabetes-related health metrics and making predictions.

## Features

### Data Exploration
- **Data Overview**: Visual distributions of all health metrics
- **Correlation Analysis**: Interactive heatmap showing feature relationships
- **Cluster Analysis**: Patient segmentation using K-means clustering

### Machine Learning
- **Model Comparison**: Random Forest vs Logistic Regression performance metrics
- **Confusion Matrices**: Detailed model evaluation visualizations
- **Live Prediction**: Interactive tool for real-time diabetes risk assessment

### Visualization
- Interactive plots with Seaborn and Matplotlib
- Dark/Light theme toggle
- Responsive UI with CustomTkinter widgets

## Technical Implementation

### Requirements
- Python 3.8+
- Required packages:
customtkinter
pandas
numpy
scikit-learn
matplotlib
seaborn

text

### Data Structure
The application processes a diabetes dataset with features including:
- HbA1c levels
- BMI
- Age
- Blood glucose levels
- Hypertension status
- Heart disease status
- Smoking history (categorized)

### Machine Learning Models
1. **Random Forest Classifier**
 - 150 estimators
 - Max depth of 7
 - Balanced class weights
 - Optimized hyperparameters

2. **Logistic Regression**
 - L2 regularization
 - Balanced class weights
 - Liblinear solver

## Installation

1. Clone the repository:
 ```bash
 git clone https://github.com/MrTsak/AI-Data-Analytics.git
 cd AI-Data-Analytics
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
python main.py
Usage Guide
Data Exploration:

Navigate through tabs to view different visualizations

Examine correlations between features

Explore patient clusters

Model Analysis:

Compare model performance metrics

View ROC curves and confusion matrices

See which model performs better

Live Prediction:

Adjust sliders for patient parameters

Select a prediction model

View risk probability and interpretation

File Structure
text
AI-Data-Analytics/
├── data/
│   └── sample.csv          # Diabetes dataset
├── main.py                 # Main application code
├── README.md               # This documentation
└── requirements.txt        # Package requirements
Key Technical Aspects
Data Preprocessing:

Handling of categorical variables (smoking history, gender)

Missing value imputation

Feature scaling for clustering

UI/UX Features:

Theme switching (dark/light mode)

Responsive layout

Interactive widgets

Performance Optimization:

Memory management

Efficient plotting

Model caching

Contributing
Contributions are welcome! Please:

Fork the repository

Create a feature branch

Submit a pull request

License
MIT License