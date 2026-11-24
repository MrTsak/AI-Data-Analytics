import * as charts from './charts.js';
import { LABELS } from './charts.js';
import { setupPredictor } from './predictor.js';

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

const state = { data: null };
const $ = (sel) => document.querySelector(sel);
const fmtPct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

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
    s.append(label);
    p.append(s);
  }
  return p;
}

// ---------- static text ----------
function renderText(d) {
  const s = d.summary;
  const best = d.models.metrics[s.best];
  document.querySelectorAll('[data-bind="n_rows"]').forEach((el) => { el.textContent = s.n_rows.toLocaleString(); });
  document.querySelectorAll('[data-bind="positive_pct"]').forEach((el) => { el.textContent = fmtPct(s.positive_rate, 1); });
  $('[data-bind="hero-lede"]').textContent =
    `${s.n_original.toLocaleString()} anonymised health records went in. After removing ${s.n_duplicates.toLocaleString()} exact duplicates, ` +
    `two models were trained to spot diabetes. The ${s.best} finds ${fmtPct(best.recall, 0)} of diabetic patients it has never seen, ` +
    `with an ROC-AUC of ${best.auc.toFixed(3)}.`;

  $('#hypotheses').replaceChildren(...HYPOTHESES.map(([text, url]) => {
    const li = document.createElement('li');
    li.append(`${text} `);
    const a = document.createElement('a');
    a.href = url; a.textContent = 'source'; a.rel = 'noopener'; a.target = '_blank';
    li.append('(', a, ')');
    return li;
  }));

  const c = charts.palette();
  $('#cluster-list').replaceChildren(...d.clusters.descriptions.map((text, i) => {
    const li = document.createElement('li');
    li.style.setProperty('--c', c.clusters[i]);
    const [tag, rest] = text.split(': ');
    const strong = document.createElement('strong');
    strong.textContent = `Group ${i + 1}: ${tag.toLowerCase()}`;
    const span = document.createElement('span');
    span.textContent = `${d.clusters.sizes[i].toLocaleString()} patients. ${rest.charAt(0).toUpperCase()}${rest.slice(1)}.`;
    li.append(strong, span);
    return li;
  }));

  renderMetricsTable(d.models);
  renderFindings(d);
}

function renderMetricsTable(models) {
  const rows = [['auc', 'ROC-AUC'], ['recall', 'Recall'], ['precision', 'Precision'], ['f1', 'F1 score'], ['accuracy', 'Accuracy']];
  const head = `<thead><tr><th scope="col">Metric</th>${MODEL_NAMES.map((n) => `<th scope="col">${n}</th>`).join('')}</tr></thead>`;
  const body = rows.map(([key, label]) => {
    const vals = MODEL_NAMES.map((n) => models.metrics[n][key]);
    const max = Math.max(...vals);
    return `<tr><th scope="row">${label}</th>${vals.map((v) => `<td class="${v === max ? 'best' : ''}">${v.toFixed(3)}</td>`).join('')}</tr>`;
  }).join('');
  $('#metrics-table').innerHTML = `${head}<tbody>${body}</tbody>`;
}

function renderFindings(d) {
  const s = d.summary;
  const m = d.models.metrics;
  const plain = { HbA1c_level: 'HbA1c', blood_glucose_level: 'blood glucose', age: 'age', bmi: 'BMI' };
  const top = Object.entries(d.models.importances).sort((a, b) => b[1] - a[1]).slice(0, 2)
    .map(([f]) => plain[f] ?? LABELS[f].toLowerCase());
  const rates = d.clusters.diabetes_rate;
  const hi = rates.indexOf(Math.max(...rates));
  const other = MODEL_NAMES.find((n) => n !== s.best);
  const items = [
    `${top[0].charAt(0).toUpperCase()}${top[0].slice(1)} and ${top[1]} carry most of the predictive signal. Together they account for most of the Random Forest's feature importance.`,
    `The ${s.best} edges out ${other} (AUC ${m[s.best].auc.toFixed(3)} vs ${m[other].auc.toFixed(3)}). Both catch about ${fmtPct(m[s.best].recall, 0)} of diabetic patients, but ${m[s.best].precision < 0.5 ? 'more than half of the patients it flags are false alarms' : `${fmtPct(1 - m[s.best].precision, 0)} of the patients it flags are false alarms`} (precision ${fmtPct(m[s.best].precision, 0)}). That trade-off suits screening, where a follow-up test settles it.`,
    `Group ${hi + 1} has a diabetes rate of ${fmtPct(rates[hi])}, against ${fmtPct(s.positive_rate)} overall. Clustering alone picks out a high-risk population.`,
    `${s.n_duplicates.toLocaleString()} duplicate records and ${s.n_other_gender} records with gender "Other" (too few to model) were removed before training, so test scores aren't inflated by repeated rows.`,
    'The dataset has no lifestyle, diet, family-history or genetic data, and the results have not been clinically validated.',
  ];
  $('#findings').replaceChildren(...items.map((t) => Object.assign(document.createElement('li'), { textContent: t })));
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
  $('#feature-note').textContent = FEATURE_NOTES[feat] ?? 'Bars show the share of patients in each category who have diabetes.';
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
  picker.replaceChildren(...options.map((f) => new Option(LABELS[f], f)));
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
  const label = () => { btn.textContent = currentTheme() === 'dark' ? 'Light' : 'Dark'; };
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
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
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
