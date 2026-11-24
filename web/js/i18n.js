// Greek text for the project page's scripts: charts, generated sentences and the predictor.
// web/el/index.html sets lang="el" and holds the page's own Greek text; the English page is unchanged.
export const isGreek = document.documentElement.lang === 'el';
export const locale = isGreek ? 'el-GR' : 'en-US';

/** Pick the English or Greek text. */
export const pick = (en, gr) => (isGreek ? gr : en);

/** Static English text to Greek. Unknown text is returned unchanged. */
const GREEK = {
  'Age (years)': 'Ηλικία (έτη)',
  Age: 'Ηλικία',
  'Blood glucose': 'Γλυκόζη αίματος',
  'Blood glucose (mg/dL)': 'Γλυκόζη αίματος (mg/dL)',
  'Heart disease': 'Καρδιοπάθεια',
  Hypertension: 'Υπέρταση',
  Gender: 'Φύλο',
  'Smoking history': 'Ιστορικό καπνίσματος',
  Diabetes: 'Διαβήτης',
  'No diabetes': 'Χωρίς διαβήτη',
  Male: 'Άνδρας',
  Female: 'Γυναίκα',
  No: 'Όχι',
  Yes: 'Ναι',
  'No info': 'Χωρίς στοιχεία',
  'Never smoked': 'Δεν έχει καπνίσει ποτέ',
  'Former smoker': 'Πρώην καπνιστής',
  'Current smoker': 'Τρέχων καπνιστής',
  'Not currently smoking': 'Δεν καπνίζει σήμερα',
  'Has smoked': 'Έχει καπνίσει',
  'Diabetes rate (%)': 'Ποσοστό διαβήτη (%)',
  '% of group': '% της ομάδας',
  'False positive rate': 'Ψευδώς θετικά (FPR)',
  'True positive rate (recall)': 'Αληθώς θετικά (ανάκληση)',
  'Feature importance': 'Σημασία χαρακτηριστικού',
  'What the Random Forest relies on': 'Πάνω σε τι βασίζεται το Τυχαίο δάσος',
  Predicted: 'Πρόβλεψη',
  Actual: 'Πραγματικό',
  'Random Forest': 'Τυχαίο δάσος',
  'Logistic Regression': 'Λογιστική παλινδρόμηση',
  // Theme button
  Dark: 'Σκούρο',
  Light: 'Φωτεινό',
  // Charts, metrics, hypotheses and feature notes
  'Higher HbA1c goes with higher diabetes risk': 'Υψηλότερο HbA1c συνδέεται με υψηλότερο κίνδυνο διαβήτη',
  'BMI is positively related to diabetes risk': 'Το BMI σχετίζεται θετικά με τον κίνδυνο διαβήτη',
  'Risk rises with age': 'Ο κίνδυνος αυξάνεται με την ηλικία',
  'Blood glucose is strongly related to diabetes': 'Η γλυκόζη αίματος σχετίζεται ισχυρά με τον διαβήτη',
  source: 'πηγή',
  'Lowest risk': 'χαμηλότερος κίνδυνος',
  'Middle risk': 'μεσαίος κίνδυνος',
  'Highest risk': 'υψηλότερος κίνδυνος',
  'Metric': 'Μέτρο',
  'ROC-AUC': 'ROC-AUC',
  Recall: 'Ανάκληση (recall)',
  Precision: 'Ακρίβεια θετικών (precision)',
  'F1 score': 'F1',
  Accuracy: 'Ακρίβεια (accuracy)',
  // Predictor
  Low: 'Χαμηλή',
  Moderate: 'Μέτρια',
  High: 'Υψηλή',
  'Risk scale': 'Κλίμακα κινδύνου',
};

/** Static English text to Greek (unchanged on the English page). */
export const t = (text) => (isGreek && GREEK[text]) || text;

/** Chart labels for features and categories. */
export const featureLabel = (key, labels) => t(labels[key] ?? key);

/** Model names as displayed on this page. */
export const modelName = (name) => t(name);

/** Format a number with the page's decimal separator. */
export const num = (x, digits = 0) =>
  x.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits });

/** Format a fraction (0.123) as a percentage with the page's separator. */
export const pct = (fraction, digits = 1) => `${num(fraction * 100, digits)}%`;
