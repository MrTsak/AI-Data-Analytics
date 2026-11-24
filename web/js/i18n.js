// Greek text for the project page. web/el/index.html sets lang="el"; the English page is unchanged.
// Predictor labels and results stay in English on the Greek page (see web/el/index.html).
export const isGreek = document.documentElement.lang === 'el';
export const locale = isGreek ? 'el-GR' : 'en-US';

/** Pick the English or Greek text. */
export const pick = (en, gr) => (isGreek ? gr : en);

/** Static English text to Greek. Unknown text is returned unchanged. */
const GREEK = {
  'Age (years)': 'Ηλικία (έτη)',
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
  // Page text
  'Diabetes Risk Explorer': 'Εξερευνητής κινδύνου διαβήτη',
  'Skip to content': 'Μετάβαση στο περιεχόμενο',
  Explore: 'Εξερεύνηση',
  Clusters: 'Ομάδες',
  Models: 'Μοντέλα',
  'Try it': 'Δοκίμασε',
  Dark: 'Σκούρο',
  Light: 'Φωτεινό',
  'Switch colour theme': 'Αλλαγή θέματος χρωμάτων',
  'Try the risk estimator': 'Δοκίμασε τον εκτιμητή κινδύνου',
  'Read the code on GitHub': 'Δες τον κώδικα στο GitHub',
  'Educational project — not a medical device or diagnosis.':
    'Εκπαιδευτικό έργο — δεν είναι ιατρική συσκευή ούτε διάγνωση.',
  'Explore the data': 'Εξερεύνηση των δεδομένων',
  'Pick a measurement to compare patients with and without diabetes. Continuous values are shown as a share of each group, so the smaller diabetic group stays visible.':
    'Επίλεξε μια μέτρηση για να συγκρίνεις ασθενείς με και χωρίς διαβήτη. Οι συνεχείς τιμές εμφανίζονται ως ποσοστό κάθε ομάδας, ώστε η μικρότερη ομάδα διαβητικών να φαίνεται καθαρά.',
  Measurement: 'Μέτρηση',
  'How the measurements relate': 'Πώς σχετίζονται οι μετρήσεις',
  'Pearson correlation between each pair of measurements and the diabetes outcome. The bottom row is the one to read first.':
    'Συντελεστής συσχέτισης Pearson για κάθε ζεύγος μετρήσεων και για την έκβαση του διαβήτη. Η κάτω γραμμή είναι αυτή που πρέπει να διαβάσεις πρώτη.',
  'Patient groups': 'Ομάδες ασθενών',
  'K-means clustering on age, BMI and HbA1c finds three groups without being told who has diabetes. The diabetes rate per group is measured afterwards.':
    'Η ομαδοποίηση K-means με βάση την ηλικία, το BMI και το HbA1c βρίσκει τρεις ομάδες χωρίς να του δοθεί ποιος έχει διαβήτη. Το ποσοστό διαβήτη κάθε ομάδας μετριέται στη συνέχεια.',
  'Comparing the models': 'Σύγκριση των μοντέλων',
  'Move the sliders and both models update instantly. They run in your browser; nothing you enter leaves this page.':
    'Μετακίνησε τους ρυθμιστές και τα δύο μοντέλα ενημερώνονται αμέσως. Τρέχουν στον περιηγητή σου· ό,τι εισάγεις δεν φεύγει από τη σελίδα.',
  'Educational project — not a medical device or diagnosis. The estimates describe this dataset only, have not been clinically validated, and are not anyone\'s actual risk.':
    'Εκπαιδευτικό έργο — δεν είναι ιατρική συσκευή ούτε διάγνωση. Οι εκτιμήσεις αφορούν μόνο αυτό το σύνολο δεδομένων, δεν έχουν επικυρωθεί κλινικά και δεν είναι ο πραγματικός κίνδυνος κανενός.',
  'How it was tested': 'Πώς δοκιμάστηκε',
  'Findings and limitations': 'Ευρήματα και περιορισμοί',
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
