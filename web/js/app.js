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
  HbA1c_level: 'HbA1c of 6.5% or above is a standard diagnostic threshold, which is why the diabetic group sits to the right.',
  blood_glucose_level: 'Diabetic patients cluster at higher glucose readings, with a long tail above 200 mg/dL.',
  bmi: 'The many patients at exactly 27.32 are records where BMI was filled in with the dataset average.',
  age: 'Diabetes is rare under 40 in this dataset and rises steadily with age.',
};
const FEATURE_NOTES_EL = {
  HbA1c_level: 'Η τιμή HbA1c 6,5% ή άνω είναι τυπικό διαγνωστικό όριο, γι’ αυτό η ομάδα με διαβήτη βρίσκεται δεξιότερα.',
  blood_glucose_level: 'Οι διαβητικοί ασθενείς συγκεντρώνονται σε υψηλότερες τιμές γλυκόζης, με μακριά ουρά πάνω από 200 mg/dL.',
  bmi: 'Οι πολλοί ασθενείς με ακριβώς 27,32 είναι εγγραφές όπου το BMI συμπληρώθηκε με τον μέσο όρο του συνόλου δεδομένων.',
  age: 'Ο διαβήτης είναι σπάνιος κάτω των 40 ετών σε αυτό το σύνολο και αυξάνεται σταθερά με την ηλικία.',
};
const DEFAULT_NOTE = pick(
  'Bars show the share of patients in each category who have diabetes.',
  'Οι μπάρες δείχνουν το ποσοστό των ασθενών σε κάθε κατηγορία που έχουν διαβήτη.',
);

const state = { data: null };
const $ = (sel) => document.querySelector(sel);
const cap = (s) => `${s.charAt(0).toUpperCase()}${s.slice(1)}`;

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

function mount(el, node) {
  el.replaceChildren(node);
}

function legend(items) {
  const p = document.createElement('p');
  p.className = 'legend';
  for (const [label, color] of items) {
    const s = document.createElement('span');
    s.innerHTML = `<i class="swatch" style="background:${color}"></i>`;
    s.append(t(label));
    p.append(s);
  }
  return p;
}

// The cluster descriptions are English sentences built from clusters.json. On the Greek page the
// same figures are reused and the sentence is rebuilt in Greek.
function clusterRest(rest) {
  if (!isGreek) return rest;
  const m = rest.match(/avg age ([\d.]+), BMI ([\d.]+), HbA1c ([\d.]+)% — diabetes rate ([\d.]+)%/);
  if (!m) return rest;
  const dec = (v) => v.replace('.', ',');
  return `μέση ηλικία ${m[1]}, BMI ${dec(m[2])}, HbA1c ${dec(m[3])}% — ποσοστό διαβήτη ${dec(m[4])}%`;
}

// ---------- static text ----------
function renderText(d) {
  const s = d.summary;
  const best = d.models.metrics[s.best];
  document.querySelectorAll('[data-bind="n_rows"]').forEach((el) => { el.textContent = num(s.n_rows); });
  document.querySelectorAll('[data-bind="positive_pct"]').forEach((el) => { el.textContent = pct(s.positive_rate, 1); });
  $('[data-bind="hero-lede"]').textContent = pick(
    `${num(s.n_original)} records went in. After removing ${num(s.n_duplicates)} exact duplicates, ` +
      `two models were trained to spot diabetes. The ${s.best} finds ${pct(best.recall, 0)} of diabetic patients it has never seen, ` +
      `with an ROC-AUC of ${num(best.auc, 3)}.`,
    `Μπήκαν ${num(s.n_original)} εγγραφές. Αφού αφαιρέθηκαν ${num(s.n_duplicates)} ακριβώς διπλές, ` +
      `εκπαιδεύτηκαν δύο μοντέλα για τον εντοπισμό του διαβήτη. Το καλύτερο μοντέλο («${modelName(s.best)}») ` +
      `εντοπίζει το ${pct(best.recall, 0)} των διαβητικών ασθενών που δεν έχει ξαναδεί, με ROC-AUC ${num(best.auc, 3)}.`,
  );

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
    const tagText = isGreek ? t(tag) : tag.toLowerCase();
    strong.textContent = `${pick('Group', 'Ομάδα')} ${i + 1}: ${tagText}`;
    const span = document.createElement('span');
    span.textContent = `${num(d.clusters.sizes[i])} ${pick('patients', 'ασθενείς')}. ${cap(clusterRest(rest))}.`;
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
      `${cap(top[0])} and ${top[1]} carry most of the predictive signal. Together they account for most of the Random Forest's feature importance.`,
      `Τα δύο σημαντικότερα χαρακτηριστικά είναι ${top[0]} και ${top[1]}. Μαζί εξηγούν το μεγαλύτερο μέρος της σημασίας χαρακτηριστικών του Τυχαίου δάσους.`,
    ),
    pick(
      `The ${s.best} edges out ${other} (AUC ${num(m[s.best].auc, 3)} vs ${num(m[other].auc, 3)}). It finds ${pct(m[s.best].recall, 0)} of diabetic patients, and ${pct(m[s.best].precision, 0)} of the patients it flags are diabetic. ${other} finds ${pct(m[other].recall, 0)}, with ${pct(m[other].precision, 0)} of its flags diabetic. ${weakRecall ? 'Each model misses a sizable share of diabetic patients, so neither is suitable for screening.' : 'Both models find most diabetic patients.'}`,
      `Το μοντέλο «${modelName(s.best)}» ξεπερνά ελαφρά το μοντέλο «${modelName(other)}» (AUC ${num(m[s.best].auc, 3)} έναντι ${num(m[other].auc, 3)}). ` +
        `Εντοπίζει το ${pct(m[s.best].recall, 0)} των διαβητικών ασθενών, και το ${pct(m[s.best].precision, 0)} όσων επισημαίνει είναι διαβητικοί. ` +
        `Το «${modelName(other)}» εντοπίζει το ${pct(m[other].recall, 0)}, και το ${pct(m[other].precision, 0)} των επισημάνσεών του είναι διαβητικές. ` +
        (weakRecall
          ? 'Κάθε μοντέλο χάνει ένα σημαντικό μέρος των διαβητικών ασθενών, οπότε κανένα δεν είναι κατάλληλο για έλεγχο.'
          : 'Και τα δύο μοντέλα εντοπίζουν τους περισσότερους διαβητικούς ασθενείς.'),
    ),
    pick(
      `Group ${hi + 1} has a diabetes rate of ${pct(rates[hi])}, against ${pct(s.positive_rate)} overall. Clustering alone picks out a high-risk population.`,
      `Η ομάδα ${hi + 1} έχει ποσοστό διαβήτη ${pct(rates[hi])}, έναντι ${pct(s.positive_rate)} συνολικά. Η ομαδοποίηση από μόνη της εντοπίζει έναν πληθυσμό υψηλού κινδύνου.`,
    ),
    pick(
      `${num(s.n_duplicates)} duplicate records and ${s.n_other_gender} records with gender "Other" (too few to model) were removed before training, so test scores aren't inflated by repeated rows.`,
      `Αφαιρέθηκαν ${num(s.n_duplicates)} διπλές εγγραφές και ${s.n_other_gender} εγγραφές με φύλο «Άλλο» (πολύ λίγες για μοντελοποίηση) πριν από την εκπαίδευση, ώστε οι βαθμολογίες δοκιμής να μην φουσκώνουν από επαναλαμβανόμενες γραμμές.`,
    ),
    pick(
      'The dataset has no lifestyle, diet, family-history or genetic data, and the results have not been clinically validated.',
      'Το σύνολο δεδομένων δεν περιέχει στοιχεία τρόπου ζωής, διατροφής, οικογενειακού ιστορικού ή γονιδιακά στοιχεία, και τα αποτελέσματα δεν έχουν επικυρωθεί κλινικά.',
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
    fig.append(legend([['No diabetes', c.neg], ['Diabetes', c.pos]]), charts.histogram(d.explore.histograms[feat], feat, w));
  } else {
    fig.append(charts.categoryRates(d.explore.categories[feat], feat, w));
  }
  el.replaceChildren(fig);
  const notes = isGreek ? FEATURE_NOTES_EL : FEATURE_NOTES;
  $('#feature-note').textContent = notes[feat] ?? DEFAULT_NOTE;
}

function renderCharts() {
  const d = state.data;
  renderExplore();
  mount($('#corr-chart'), charts.correlationHeatmap(d.explore.correlation, widthOf($('#corr-chart'))));
  mount($('#cluster-chart'), charts.clusterScatter(d.clusters.grid, widthOf($('#cluster-chart'))));
  const stack = $('#roc-chart');
  mount(stack, charts.rocChart(d.models.metrics, MODEL_NAMES, widthOf(stack)));
  const conf = $('#confusion-charts');
  const cw = conf.getBoundingClientRect().width;
  const each = cw < 480 ? cw : (cw - 24) / 2;
  conf.replaceChildren(...MODEL_NAMES.map((n) => charts.confusion(d.models.metrics[n].confusion, n, Math.max(260, each))));
  mount($('#importance-chart'), charts.importanceBars(d.models.importances, widthOf($('#importance-chart'))));
  // cluster list colours follow the theme
  const c = charts.palette();
  document.querySelectorAll('#cluster-list li').forEach((li, i) => li.style.setProperty('--c', c.clusters[i]));
}

function setupPicker() {
  const picker = $('#feature-picker');
  const options = ['HbA1c_level', 'blood_glucose_level', 'bmi', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'gender'];
  picker.replaceChildren(...options.map((f) => new Option(featureLabel(f, LABELS), f)));
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
  const label = () => { btn.textContent = currentTheme() === 'dark' ? t('Light') : t('Dark'); };
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
