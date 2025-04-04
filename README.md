# Diabetes Prediction Dashboard

## English

### 📌 Overview
A Python-based interactive dashboard for diabetes prediction analysis using machine learning (Random Forest and Logistic Regression) with a modern GUI built with CustomTkinter.

### ✨ Features
- **Data Analysis**: Explore diabetes dataset with statistical visualizations
- **Machine Learning**: Two trained models (Random Forest and Logistic Regression)
- **Clustering**: K-means clustering analysis (3 clusters)
- **Interactive GUI**: Modern interface with light/dark mode toggle
- **Comprehensive Visualizations**: 
  - Correlation heatmaps
  - Feature importance charts
  - ROC curves
  - Confusion matrices
  - Cluster visualizations

### 📊 Dataset
The dataset contains clinical parameters for diabetes prediction:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (diabetes diagnosis)

### ⚙️ Requirements
```
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
customtkinter
```

### 🚀 Installation
1. Clone the repository
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Place your diabetes dataset in `data/sample.csv`
4. Run:
```bash
python main.py
```

### 🏗️ Project Structure
```
.
├── data/               # Dataset directory
│   └── sample.csv      # Diabetes dataset
├── main.py  # Main application
└── README.md           # This file
```

### 📋 Key Metrics
- Random Forest Accuracy: ~80%
- Top Predictive Features:
  1. Glucose
  2. BMI
  3. Age
  4. Diabetes Pedigree Function
---

## Ελληνικά

### 📌 Επισκόπηση
Μια διαδραστική πλατφόρμα ανάλυσης και πρόβλεψης διαβήτη με Python, μηχανική μάθηση (Random Forest και Logistic Regression) και σύγχρονο γραφικό περιβάλλον (CustomTkinter).

### ✨ Χαρακτηριστικά
- **Ανάλυση Δεδομένων**: Εξερεύνηση δεδομένων με στατιστικές απεικονίσεις
- **Μηχανική Μάθηση**: 2 εκπαιδευμένα μοντέλα
- **Ομαδοποίηση**: Ανάλυση με K-means (3 clusters)
- **Διαδραστική Διεπαφή**: Σύγχρονο γραφικό περιβάλλον με εναλλαγή θέματος
- **Οπτικοποιήσεις**:
  - Χάρτες συσχέτισης
  - Διαγράμματα σημαντικότητας χαρακτηριστικών
  - Καμπύλες ROC
  - Πίνακες σύγχυσης
  - Απεικονίσεις clusters

### 📊 Δεδομένα
Το dataset περιέχει κλινικές παραμέτρους:
- Εγκυμοσύνες
- Γλυκόζη
- Πίεση αίματος
- Πάχος δέρματος
- Ινσουλίνη
- BMI
- Συνάρτηση γενετικής προδιάθεσης
- Ηλικία
- Διάγνωση διαβήτη

### ⚙️ Απαιτήσεις
```
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
customtkinter
```

### 🚀 Εγκατάσταση
1. Κλωνοποίηση του repository
2. Εγκατάσταση απαιτήσεων:
```bash
pip install -r requirements.txt
```
3. Τοποθέτηση dataset στο `data/sample.csv`
4. Εκτέλεση:
```bash
python main.py
```

### 📋 Βασικά Μεγέθη
- Ακρίβεια Random Forest: ~80%
- Σημαντικότερα χαρακτηριστικά:
  1. Γλυκόζη
  2. BMI
  3. Ηλικία
  4. Συνάρτηση γενετικής προδιάθεσης

### 🏆 Σημαντικές Παρατηρήσεις
- Η γλυκόζη είναι ο πιο σημαντικός δείκτης
- Το μοντέλο μπορεί να βοηθήσει στην έγκαιρη διάγνωση
- Οι ομάδες ασθενών (clusters) έχουν διαφορετικά προφίλ κινδύνου