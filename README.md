# 🩺 Diabetes Prediction Dashboard

## 🌟 Overview

A comprehensive **machine learning dashboard** for analyzing diabetes risk factors and predicting outcomes. Provides interactive visualizations of clinical data and model performance metrics with a beautiful modern interface.

## ✨ Features

### 📊 Data Analysis
| Feature | Description |
|---------|-------------|
| 📈 Exploratory Visualizations | Histograms and distribution plots for all features |
| 🔥 Correlation Heatmap | Interactive correlation matrix visualization |
| 🎨 Patient Clustering | K-means clustering analysis (3 clusters) |

### 🤖 Machine Learning
| Model | Accuracy | Key Features |
|-------|----------|--------------|
| Random Forest | ~72.1% | Feature importance analysis |
| Logistic Regression | ~74.7% | Coefficient interpretation |
| **Model Comparison** | Side-by-side metrics | ROC curve visualization |

### 🎨 Visualization
- Interactive Matplotlib/Seaborn charts
- Confusion matrix display
- Dynamic ROC curve analysis
- Light/Dark mode toggle

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MrTsak/AI-Data-Analytics.git
   cd AI-Data-Analytics

📦 Dependencies
Package	Version
Python	≥ 3.7
customtkinter	≥ 5.2.1
pandas	≥ 1.3.0
scikit-learn	≥ 1.0.0

bash
Install dependencies:

bash
pip install -r requirements.txt
Place your dataset at data/sample.csv (semicolon-delimited)

🚀 Usage
Launch the application:

bash
python main.py
Interface Guide:
Tab Navigation: Switch between different analysis views

🎨 Theme Toggle: Click the sun/moon icon to change themes

📊 Interactive Plots: Hover for values, click to explore

🔍 Research Links: Click "[source]" for reference materials

📂 Data Format
Required CSV structure:

csv
Pregnancies;Glucose;BloodPressure;SkinThickness;Insulin;BMI;DiabetesPedigreeFunction;Age;Outcome
Example row:
csv
6;148;72;35;0;33.6;0.627;50;1

🗂 Project Structure
AI-Data-Analytics/
├── data/
│   └── sample.csv          # Dataset
├── main.py                 # Main application
├── requirements.txt        # Dependency list
└── README.md               # Documentation