// Greek text for the project page's scripts: charts, generated sentences and the predictor.
// web/el/index.html sets lang="el" and holds the page's own Greek text; the English page is unchanged.
export const isGreek = document.documentElement.lang === 'el';
export const locale = isGreek ? 'el-GR' : 'en-US';

// Chart axes are drawn by Observable Plot with d3's number format. On the Greek page, give d3 the
// Greek separators (0,5 and 1.000), so ticks match the rest of the page. Only the labels change.
if (isGreek && globalThis.d3?.formatDefaultLocale) {
  globalThis.d3.formatDefaultLocale({ decimal: ',', thousands: '.', grouping: [3], currency: ['', '\u00a0€'] });
}

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
  'Current smoker': 'Καπνίζει',
  'Not currently smoking': 'Δεν καπνίζει τώρα',
  'Has smoked': 'Έχει καπνίσει',
  'Records with diabetes (%)': 'Ποσοστό διαβήτη (%)',
  '% of group': 'Ποσοστό της ομάδας (%)',
  'False positive rate': 'Ποσοστό ψευδώς θετικών (FPR)',
  'True positive rate (recall)': 'Ποσοστό αληθώς θετικών (TPR)',
  'Feature importance': 'Σημαντικότητα μεταβλητής',
  'Which variables the Random Forest uses most': 'Ποιες μεταβλητές χρησιμοποιεί περισσότερο το Random Forest',
  'Model prediction': 'Πρόβλεψη μοντέλου',
  'Actual class': 'Πραγματική κατηγορία',
  'Random Forest': 'Random Forest',
  'Logistic Regression': 'Λογιστική παλινδρόμηση',
  // Theme button
  Dark: 'Σκούρο θέμα',
  Light: 'Φωτεινό θέμα',
  // Charts, metrics, hypotheses and feature notes
  'Higher HbA1c goes with higher diabetes risk': 'Υψηλότερες τιμές HbA1c συνδέονται με αυξημένο κίνδυνο διαβήτη.',
  'Overweight and obesity are associated with a higher risk of type 2 diabetes': 'Το υπερβολικό βάρος και η παχυσαρκία συνδέονται με αυξημένο κίνδυνο εμφάνισης διαβήτη τύπου 2.',
  'Risk rises with age': 'Ο κίνδυνος διαβήτη αυξάνεται με την ηλικία.',
  'Blood glucose is strongly related to diabetes': 'Οι τιμές γλυκόζης αίματος συνδέονται στενά με τον διαβήτη.',
  source: 'πηγή',
  'Lowest risk': 'Χαμηλότερο ποσοστό διαβήτη',
  'Middle risk': 'Ενδιάμεσο ποσοστό διαβήτη',
  'Highest risk': 'Υψηλότερο ποσοστό διαβήτη',
  'Metric': 'Δείκτης αξιολόγησης',
  'ROC-AUC': 'ROC-AUC',
  Recall: 'Ανάκληση (recall)',
  Precision: 'Ορθότητα θετικών προβλέψεων (precision)',
  'F1 score': 'F1',
  Accuracy: 'Ποσοστό σωστών προβλέψεων (accuracy)',
  // Predictor
  Low: 'Χαμηλή',
  Moderate: 'Μέτρια',
  High: 'Υψηλή',
  'Model estimate scale': 'Κλίμακα εκτίμησης του μοντέλου',
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
