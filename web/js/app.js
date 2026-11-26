import * as charts from './charts.js';
import { LABELS } from './charts.js';
import { setupPredictor } from './predictor.js';
import { t, pick, pct, num, modelName, featureLabel, isGreek } from './i18n.js';

const MODEL_NAMES = ['Random Forest', 'Logistic Regression'];
const FILES = ['summary', 'explore', 'clusters', 'models', 'model_lr', 'model_rf', 'form'];
const HYPOTHESES = [
  ['Higher HbA1c goes with higher diabetes risk', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4933534/'],
  ['BMI is positively related to diabetes risk', 'https://www.escardio.org/The-ESC/Press-Office/Press-releases/Body-mass-index-is-a-more-powerful-risk-factor-for-diabetes-than-genetics'],
  ['Risk rises with age', 'https://www.thelancet.com/journals/lanhl/article/PIIS2666-7568(21)00177-X/fulltext'],
  ['Blood glucose is strongly related to diabetes', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4484145/'],
];
const FEATURE_NOTES = {
  HbA1c_level: 'In this dataset, records with diabetes appear mostly at higher HbA1c values.',
  blood_glucose_level: 'Records with diabetes appear more often at high glucose values, and the distribution extends above 200 mg/dL.',
  bmi: 'Many records have exactly the same BMI value: 27.32.',
  age: 'In this dataset, few records under age 40 have diabetes, and the share rises with age.',
};
const FEATURE_NOTES_EL = {
  HbA1c_level: 'Στο συγκεκριμένο σύνολο δεδομένων, οι εγγραφές με διαβήτη εμφανίζονται κυρίως σε υψηλότερες τιμές HbA1c.',
  blood_glucose_level: 'Οι εγγραφές με διαβήτη εμφανίζονται συχνότερα σε υψηλές τιμές γλυκόζης, με την κατανομή να εκτείνεται και πάνω από τα 200 mg/dL.',
  bmi: 'Πολλές εγγραφές έχουν ακριβώς την ίδια τιμή BMI: 27,32.',
  age: 'Στο συγκεκριμένο σύνολο δεδομένων, οι εγγραφές με διαβήτη είναι λιγότερες στις ηλικίες κάτω των 40 ετών και το ποσοστό τους αυξάνεται με την ηλικία.',
};
const DEFAULT_NOTE = pick(
  'Bars show the share of records with diabetes in each category.',
  'Οι στήλες δείχνουν το ποσοστό των εγγραφών με διαβήτη σε κάθε κατηγορία.',
);

const state = { data: null };
const $ = (sel) => document.querySelector(sel);
const cap = (s) => `${s.charAt(0).toUpperCase()}${s.slice(1)}`;
// English model names inside sentences, and the English group labels for the cluster tags in clusters.json.
const EN_MODEL = { 'Random Forest': 'the Random Forest', 'Logistic Regression': 'logistic regression' };
const EN_GROUP = {
  'Lowest risk': 'lowest proportion of records with diabetes',
  'Middle risk': 'middle proportion of records with diabetes',
  'Highest risk': 'highest proportion of records with diabetes',
};
// Greek forms of the model names (subject / after a preposition).
const EL_MODEL = {
  'Random Forest': { nom: 'Το Random Forest', acc: 'το Random Forest' },
  'Logistic Regression': { nom: 'Η λογιστική παλινδρόμηση', acc: 'τη λογιστική παλινδρόμηση' },
};
// The page ships with these texts already filled in; rewriting identical text would repaint the
// hero and count as a late largest paint, so only changed text is written.
const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };

async function loadData() {
  const entries = await Promise.all(FILES.map(async (name) => {
    const res = await fetch(new URL(`../data/${name}.json`, import.meta.url));
    if (!res.ok) throw new Error(`${name}.json: HTTP ${res.status}`);
    return [name, await res.json()];
  }));
  return Object.fromEntries(entries);
}

function widthOf(el) {
  return Math.max(280, Math.floor(el.getBoundingClientRect().width));
}

function mount(el, ...nodes) {
  el.replaceChildren(...nodes);
}

// Legend keys draw the same cue as the chart: bar fill, outline, line style or dot shape.
const SWATCH = {
  solid: (c) => `<i class="swatch" style="background:${c};opacity:.75"></i>`,
  outline: (c) => `<i class="swatch" style="border:2px solid ${c};background:color-mix(in srgb, ${c} 30%, transparent)"></i>`,
  line: (c) => `<svg class="key" viewBox="0 0 26 10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="${c}" stroke-width="2.5"/></svg>`,
  dash: (c) => `<svg class="key" viewBox="0 0 26 10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" stroke="${c}" stroke-width="2.5" stroke-dasharray="7 4"/></svg>`,
  circle: (c) => `<svg class="key key-dot" viewBox="0 0 12 12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="${c}"/></svg>`,
  square: (c) => `<svg class="key key-dot" viewBox="0 0 12 12" aria-hidden="true"><rect x="1.5" y="1.5" width="9" height="9" fill="${c}"/></svg>`,
  triangle: (c) => `<svg class="key key-dot" viewBox="0 0 12 12" aria-hidden="true"><path d="M6 .8 11.4 10.6H.6z" fill="${c}"/></svg>`,
};

function legend(items) {
  const p = document.createElement('p');
  p.className = 'legend';
  for (const [label, color, kind = 'solid'] of items) {
    const s = document.createElement('span');
    s.innerHTML = SWATCH[kind](color);
    s.append(typeof label === 'string' ? t(label) : label.text);
    p.append(s);
  }
  return p;
}

// The cluster descriptions are English sentences built from clusters.json. On the Greek page the
// same figures are reused and the sentence is rebuilt in Greek.
function clusterRest(rest) {
  const m = rest.match(/avg age ([\d.]+), BMI ([\d.]+), HbA1c ([\d.]+)% — diabetes rate ([\d.]+)%/);
  if (!m) return rest;
  if (!isGreek) return `Average age: ${m[1]} years · Average BMI: ${m[2]} · Average HbA1c: ${m[3]}% · Records with diabetes: ${m[4]}%`;
  const dec = (v) => v.replace('.', ',');
  return `Μέση ηλικία: ${m[1]} έτη · Μέσο BMI: ${dec(m[2])} · Μέση HbA1c: ${dec(m[3])}% · Ποσοστό εγγραφών με διαβήτη: ${dec(m[4])}%`;
}

// ---------- static text ----------
function renderText(d) {
  const s = d.summary;
  const best = d.models.metrics[s.best];
  document.querySelectorAll('[data-bind="n_rows"]').forEach((el) => setText(el, num(s.n_rows)));
  document.querySelectorAll('[data-bind="positive_pct"]').forEach((el) => setText(el, pct(s.positive_rate, 1)));
  setText($('[data-bind="hero-lede"]'), pick(
    `This educational project starts from an initial set of ${num(s.n_original)} records. After removing ${num(s.n_duplicates)} duplicate records ` +
      `and ${num(s.n_other_gender)} records with “Other” in the gender field, ${num(s.n_rows)} records were used. Two machine-learning models were ` +
      `trained and compared. On the test set, ${EN_MODEL[s.best]} reached an ROC-AUC of ${num(best.auc, 3)} and identified ${pct(best.recall, 0)} of the ` +
      `records with diabetes.`,
    `Σε αυτό το εκπαιδευτικό έργο ανέλυσα ένα αρχικό σύνολο ${num(s.n_original)} εγγραφών. Μετά την αφαίρεση ${num(s.n_duplicates)} διπλότυπων ` +
      `εγγραφών και ${num(s.n_other_gender)} εγγραφών με τιμή «Άλλο» στο πεδίο φύλου, χρησιμοποιήθηκαν ${num(s.n_rows)} εγγραφές. Εκπαιδεύτηκαν και ` +
      `συγκρίθηκαν δύο μοντέλα μηχανικής μάθησης. Στο σύνολο δοκιμής, το ${s.best === 'Random Forest' ? 'Random Forest (τυχαίο δάσος)' : 'μοντέλο λογιστικής παλινδρόμησης'} πέτυχε ROC-AUC ${num(best.auc, 3)} ` +
      `και εντόπισε το ${pct(best.recall, 0)} των εγγραφών με διαβήτη.`,
  ));

  $('#hypotheses').replaceChildren(...HYPOTHESES.map(([text, url]) => {
    const li = document.createElement('li');
    li.append(`${t(text)} `);
    const a = document.createElement('a');
    a.href = url; a.textContent = t('source'); a.rel = 'noopener'; a.target = '_blank';
    li.append('(', a, ')');
    return li;
  }));

  const c = charts.palette();
  $('#cluster-list').replaceChildren(...d.clusters.descriptions.map((text, i) => {
    const li = document.createElement('li');
    li.style.setProperty('--c', c.clusters[i]);
    const [tag, rest] = text.split(': ');
    const strong = document.createElement('strong');
    strong.textContent = isGreek ? `Ομάδα ${i + 1} — ${t(tag)}` : `Group ${i + 1}: ${EN_GROUP[tag] ?? tag.toLowerCase()}`;
    const span = document.createElement('span');
    span.textContent = isGreek
      ? `${num(d.clusters.sizes[i])} εγγραφές · ${clusterRest(rest)}.`
      : `${num(d.clusters.sizes[i])} records · ${clusterRest(rest)}.`;
    li.append(strong, span);
    return li;
  }));

  renderMetricsTable(d.models);
  renderFindings(d);
}

function renderMetricsTable(models) {
  const rows = [['auc', 'ROC-AUC'], ['recall', 'Recall'], ['precision', 'Precision'], ['f1', 'F1 score'], ['accuracy', 'Accuracy']];
  const head = `<thead><tr><th scope="col">${t('Metric')}</th>${MODEL_NAMES.map((n) => `<th scope="col">${modelName(n)}</th>`).join('')}</tr></thead>`;
  const body = rows.map(([key, label]) => {
    const vals = MODEL_NAMES.map((n) => models.metrics[n][key]);
    const max = Math.max(...vals);
    return `<tr><th scope="row">${t(label)}</th>${vals.map((v) => `<td class="${v === max ? 'best' : ''}">${num(v, 3)}</td>`).join('')}</tr>`;
  }).join('');
  $('#metrics-table').innerHTML = `${head}<tbody>${body}</tbody>`;
}

function renderFindings(d) {
  const s = d.summary;
  const m = d.models.metrics;
  const plain = isGreek
    ? { HbA1c_level: 'HbA1c', blood_glucose_level: 'γλυκόζη αίματος', age: 'ηλικία', bmi: 'BMI' }
    : { HbA1c_level: 'HbA1c', blood_glucose_level: 'blood glucose', age: 'age', bmi: 'BMI' };
  const topName = (f) => plain[f] ?? (isGreek ? featureLabel(f, LABELS) : LABELS[f].toLowerCase());
  const top = Object.entries(d.models.importances).sort((a, b) => b[1] - a[1]).slice(0, 2)
    .map(([f]) => topName(f));
  const rates = d.clusters.diabetes_rate;
  const hi = rates.indexOf(Math.max(...rates));
  const other = MODEL_NAMES.find((n) => n !== s.best);
  const weakRecall = Math.max(m[s.best].recall, m[other].recall) < 0.9;
  const items = [
    pick(
      `The two most important variables are ${top[0]} and ${top[1]}. Together they account for most of the Random Forest’s feature importance.`,
      `Οι δύο σημαντικότερες μεταβλητές είναι: ${top[0]} και ${top[1]}. Μαζί εξηγούν το μεγαλύτερο μέρος της σημαντικότητας των μεταβλητών στο Random Forest.`,
    ),
    pick(
      `${cap(EN_MODEL[s.best])} reached a slightly higher ROC-AUC than ${EN_MODEL[other]}: ${num(m[s.best].auc, 3)} vs ${num(m[other].auc, 3)}. ` +
        `On the test set it identified ${pct(m[s.best].recall, 0)} of the records with diabetes, against about ${pct(m[other].recall, 0)} for ${EN_MODEL[other]}. ` +
        (weakRecall
          ? 'Both models missed a substantial number of positive cases, so neither is suitable for medical screening.'
          : 'Both models identify most positive cases.') +
        (m[s.best].precision === 1
          ? ` On this test set, every record that ${EN_MODEL[s.best]} classified as positive belonged to the diabetes class. That does not guarantee the same performance on new data.`
          : ` Of the records that ${EN_MODEL[s.best]} classified as positive, ${pct(m[s.best].precision, 0)} belonged to the diabetes class.`),
      `${EL_MODEL[s.best].nom} πέτυχε ελαφρώς υψηλότερο ROC-AUC από ${EL_MODEL[other].acc}: ${num(m[s.best].auc, 3)} έναντι ${num(m[other].auc, 3)}. ` +
        `Στο σύνολο δοκιμής εντόπισε το ${pct(m[s.best].recall, 0)} των εγγραφών με διαβήτη, έναντι περίπου ${pct(m[other].recall, 0)} για ${EL_MODEL[other].acc}. ` +
        (weakRecall
          ? 'Και τα δύο μοντέλα άφησαν αρκετές θετικές περιπτώσεις χωρίς εντοπισμό, οπότε κανένα δεν είναι κατάλληλο για ιατρικό προσυμπτωματικό έλεγχο.'
          : 'Και τα δύο μοντέλα εντοπίζουν τις περισσότερες θετικές περιπτώσεις.') +
        (m[s.best].precision === 1
          ? ` Στο συγκεκριμένο σύνολο δοκιμής, όλες οι εγγραφές που ταξινομήθηκαν ως θετικές από ${EL_MODEL[s.best].acc} ανήκαν στην κατηγορία με διαβήτη. Αυτό δεν εγγυάται αντίστοιχη επίδοση σε νέα δεδομένα.`
          : ` Από τις εγγραφές που ταξινομήθηκαν ως θετικές από ${EL_MODEL[s.best].acc}, το ${pct(m[s.best].precision, 0)} ανήκε στην κατηγορία με διαβήτη.`),
    ),
    pick(
      `Group ${hi + 1} has the highest proportion of records with diabetes: ${pct(rates[hi])}, against ${pct(s.positive_rate)} across the whole dataset.`,
      `Η ομάδα ${hi + 1} συγκεντρώνει μεγαλύτερο ποσοστό εγγραφών με διαβήτη: ${pct(rates[hi])}, έναντι ${pct(s.positive_rate)} στο σύνολο των δεδομένων.`,
    ),
    pick(
      `Before training, ${num(s.n_duplicates)} duplicate records were removed so that repeated rows could not inflate the test scores. A further ${s.n_other_gender} records with “Other” in the gender field were removed, as there were too few to model. This limits what the models cover and should be kept in mind when interpreting the results.`,
      `Πριν από την εκπαίδευση αφαιρέθηκαν οι διπλότυπες εγγραφές (${num(s.n_duplicates)}). Αφαιρέθηκαν επίσης ${s.n_other_gender} εγγραφές με τιμή «Άλλο» στο πεδίο φύλου· αυτή η επιλογή περιορίζει την κάλυψη του μοντέλου και πρέπει να λαμβάνεται υπόψη στην ερμηνεία των αποτελεσμάτων.`,
    ),
    pick(
      'The dataset has no information on lifestyle, diet, family history or genetic factors, and the results have not been clinically validated.',
      'Το σύνολο δεδομένων δεν περιλαμβάνει πληροφορίες για τον τρόπο ζωής, τη διατροφή, το οικογενειακό ιστορικό ή γενετικούς παράγοντες, και τα αποτελέσματα δεν έχουν επικυρωθεί κλινικά.',
    ),
  ];
  $('#findings').replaceChildren(...items.map((text) => Object.assign(document.createElement('li'), { textContent: text })));
}

// ---------- charts ----------
function renderExplore() {
  const d = state.data;
  const feat = $('#feature-picker').value;
  const el = $('#explore-chart');
  const w = widthOf(el);
  const c = charts.palette();
  const fig = document.createDocumentFragment();
  if (d.explore.histograms[feat]) {
    const name = featureLabel(feat, LABELS);
    fig.append(legend([['No diabetes', c.neg, 'solid'], ['Diabetes', c.pos, 'outline']]), charts.describe(charts.histogram(d.explore.histograms[feat], feat, w),
      pick(`Histogram of ${name} for records with and without diabetes`, `Ιστόγραμμα: ${name} σε εγγραφές με και χωρίς διαβήτη`)));
  } else {
    const name = featureLabel(feat, LABELS);
    fig.append(charts.describe(charts.categoryRates(d.explore.categories[feat], feat, w),
      pick(`Diabetes rate by ${name.toLowerCase()}`, `Ποσοστό διαβήτη ανά κατηγορία: ${name}`)));
  }
  el.replaceChildren(fig);
  const notes = isGreek ? FEATURE_NOTES_EL : FEATURE_NOTES;
  $('#feature-note').textContent = notes[feat] ?? DEFAULT_NOTE;
}

function renderCharts() {
  const d = state.data;
  renderExplore();
  mount($('#corr-chart'), charts.describe(charts.correlationHeatmap(d.explore.correlation, widthOf($('#corr-chart'))),
    pick('Correlation heatmap of the variables and diabetes', 'Χάρτης συσχέτισης των μετρήσεων και του διαβήτη')));
  const c = charts.palette();
  mount($('#cluster-chart'),
    legend(charts.CLUSTER_SYMBOLS.map((shape, i) => [{ text: `${pick('Group', 'Ομάδα')} ${i + 1}` }, c.clusters[i], shape])),
    charts.describe(charts.clusterScatter(d.clusters.grid, widthOf($('#cluster-chart'))),
      pick('Record groups by HbA1c and BMI', 'Ομάδες εγγραφών κατά HbA1c και BMI')));
  const stack = $('#roc-chart');
  mount(stack,
    legend(MODEL_NAMES.map((n, i) => [{ text: modelName(n) }, [c.ink, c.mid][i], i ? 'dash' : 'line'])),
    charts.describe(charts.rocChart(d.models.metrics, MODEL_NAMES, widthOf(stack)),
      pick('ROC curves of the two models', 'Καμπύλες ROC των δύο μοντέλων')));
  const conf = $('#confusion-charts');
  const cw = conf.getBoundingClientRect().width;
  const each = cw < 480 ? cw : (cw - 24) / 2;
  conf.replaceChildren(...MODEL_NAMES.map((n) => charts.describe(charts.confusion(d.models.metrics[n].confusion, n, Math.max(260, each)),
    pick(`Confusion matrix: ${modelName(n)}`, `Πίνακας σύγχυσης: ${modelName(n)}`))));
  mount($('#importance-chart'), charts.describe(charts.importanceBars(d.models.importances, widthOf($('#importance-chart'))),
    pick('Feature importance in the Random Forest', 'Σημαντικότητα μεταβλητών στο Random Forest')));
  // cluster list colours follow the theme
  document.querySelectorAll('#cluster-list li').forEach((li, i) => li.style.setProperty('--c', c.clusters[i]));
}

function setupPicker() {
  const picker = $('#feature-picker');
  const options = ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'gender'];
  const optionLabel = (f) => (isGreek && f === 'bmi' ? 'Δείκτης μάζας σώματος (BMI)' : featureLabel(f, LABELS));
  picker.replaceChildren(...options.map((f) => new Option(optionLabel(f), f)));
  picker.addEventListener('change', renderExplore);
}

// ---------- theme ----------
function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function setupTheme() {
  const btn = $('.theme-toggle');
  const label = () => {
    const text = currentTheme() === 'dark' ? t('Light') : t('Dark');
    btn.textContent = text;
    btn.setAttribute('aria-label', pick(`${text} theme`, text));
  };
  label();
  btn.addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch (e) { /* storage unavailable */ }
    label();
    renderCharts();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { label(); renderCharts(); });
}

function debounce(fn, ms) {
  let timer;
  return (...a) => { clearTimeout(timer); timer = setTimeout(() => fn(...a), ms); };
}

async function init() {
  try {
    state.data = await loadData();
  } catch (err) {
    $('#load-error').hidden = false;
    console.error(err);
    return;
  }
  setupPicker();
  setupTheme();
  renderText(state.data);
  renderCharts();
  setupPredictor(state.data);
  let lastWidth = innerWidth;
  addEventListener('resize', debounce(() => {
    if (innerWidth !== lastWidth) { lastWidth = innerWidth; renderCharts(); }
  }, 200));
  document.dispatchEvent(new CustomEvent('dashboard:ready', { detail: state.data }));
}

init();
