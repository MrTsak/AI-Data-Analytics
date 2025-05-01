# Diabetes Prediction Dashboard

## Overview

A comprehensive dashboard for analyzing diabetes risk factors and predicting outcomes using machine learning. Provides interactive visualizations of clinical data and model performance metrics.

## Features

### Data Analysis
- Exploratory data visualizations
- Correlation heatmaps
- Patient clustering (K-means)

### Machine Learning
- Random Forest classifier
- Logistic Regression
- Model performance comparison
- Feature importance analysis

### Visualization
- Interactive charts (Matplotlib/Seaborn)
- Confusion matrix display
- ROC curve analysis

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MrTsak/AI-Data-Analytics.git
   cd diabetes-predictor
Install dependencies:

bash
pip install -r requirements.txt
Place your dataset at data/sample.csv (semicolon-delimited format)

Usage
Run the application:

bash
python main.py
Interface Guide:
Navigate through tabs using the top menu

Toggle light/dark mode with the button in header

All visualizations are interactive

Click "[source]" links to view research references

Data Format
Required CSV columns:

Pregnancies;Glucose;BloodPressure;SkinThickness;Insulin;BMI;DiabetesPedigreeFunction;Age;Outcome
Dependencies
Python 3.7+

customtkinter
pandas
numpy
matplotlib
seaborn
scikit-learn

Project Structure
AI-DATA-ANALYTICS/
├── data/sample.csv   # Data files
├── main.py           # Main application
├── requirements.txt  # Dependencies
└── README.md         # This file