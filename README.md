# Diabetes Prediction Dashboard

An educational desktop application that combines exploratory data analysis, machine-learning model comparison, patient clustering, and interactive diabetes-risk estimation.

The application is built with Python and CustomTkinter and analyses a dataset containing 100,000 health records. It is intended as a learning and portfolio project. It is **not a medical device and must not be used for clinical diagnosis or treatment decisions**.

## Features

### Data exploration

- Visual distributions of the dataset's health metrics
- Correlation heatmap for numeric features
- Linked research hypotheses about major diabetes risk factors
- Descriptive statistics and dataset summaries

### Machine learning

- Random Forest and Logistic Regression classifiers
- Stratified 80/20 train-test split
- Balanced class weights to account for target-class imbalance
- Model comparison using accuracy, precision, recall, and F1-score
- Confusion-matrix visualisations
- ROC curves and area-under-the-curve (AUC) comparison
- Interactive risk estimation using either trained classifier

### Clustering

- K-means clustering with three patient groups
- Standardisation of age, BMI, and HbA1c features before clustering
- Cluster-centre and risk-profile visualisations

### Interface and visualisation

- Desktop interface built with CustomTkinter
- Matplotlib and Seaborn charts embedded in the application
- Light and dark appearance modes
- Scrollable tab-based interface
- Cleanup of previous charts and scheduled UI callbacks when switching tabs

## Dataset

The included CSV file contains 100,000 records and the following fields:

- Gender
- Age
- Hypertension status
- Heart-disease status
- Smoking history
- BMI
- HbA1c level
- Blood-glucose level
- Diabetes outcome

The application converts categorical gender and smoking-history values to numeric representations for modelling. Numeric missing values, if present in a replacement dataset, are filled using the corresponding column mean.

## Models

### Random Forest Classifier

- 150 estimators
- Maximum tree depth of 7
- Minimum of 8 samples required to split an internal node
- Balanced class weights
- Fixed random seed for reproducibility

### Logistic Regression

- L2 regularisation
- `liblinear` solver
- Regularisation strength `C=0.1`
- Balanced class weights
- Maximum of 1,000 iterations
- Fixed random seed for reproducibility

The application reports both models' evaluation metrics and identifies the model with the higher test-set accuracy. The displayed results describe performance on the included dataset only and do not establish clinical validity.

## Installation

Python 3.8 or newer is recommended.

```bash
git clone https://github.com/MrTsak/AI-Data-Analytics.git
cd AI-Data-Analytics
pip install -r requirements.txt
```

## Running the application

```bash
python main.py
```

The application expects the dataset at `data/sample.csv` relative to the project directory.

## Main views
