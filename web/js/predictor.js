// Live risk estimator: builds the form and updates both models on every input.
import { predictLR, predictRF, vectorize, riskBand, clampInput, explainLR, stepAlign } from './models.js';
import { LABELS } from './charts.js';

const SLIDERS = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'];
const UNITS = { age: 'years', bmi: 'kg/m²', HbA1c_level: '%', blood_glucose_level: 'mg/dL' };
const SHORT = { age: 'Age', bmi: 'BMI', HbA1c_level: 'HbA1c', blood_glucose_level: 'Blood glucose' };
const BAND_VAR = { Low: '--neg', Moderate: '--mid', High: '--pos' };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

function sliderRow(feat, r, onChange) {
  const id = `in-${feat}`;
  const { lo, hi, start } = stepAlign(r);
  const range = el('input', { type: 'range', id: `${id}-range`, min: lo, max: hi, step: r.step, value: start,
    'aria-label': `${SHORT[feat]} slider` });
  // No min on the number box: browsers step from min, and the data minimum isn't a round number.
  const number = el('input', { type: 'number', id, step: r.step, value: start, inputmode: 'decimal' });
  range.addEventListener('input', () => { number.value = range.value; onChange(); });
  number.addEventListener('input', () => {
    range.value = clampInput(number.value, r);
    onChange();
  });
  number.addEventListener('change', () => { number.value = clampInput(number.value, r); onChange(); });
  const label = el('label', { for: id }, SHORT[feat], ' ', el('small', {}, UNITS[feat]));
  return { node: el('div', { class: 'row' }, el('div', { class: 'row-head' }, label, number), range), number, range };
}

function formatProb(p) {
  if (p >= 0.995) return '>99%';
  if (p < 0.001) return '<0.1%';
  return `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
}

export function setupPredictor(data) {
  const form = document.getElementById('predict-form');
  const result = document.getElementById('predict-result');
  const { numeric, smoking } = data.form;
  const inputs = {};
  const update = () => render();

  // Gender
  const gender = el('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'Gender' },
    ...['Female', 'Male'].map((g, i) => {
      const radio = el('input', { type: 'radio', name: 'gender', value: g });
      if (i === 0) radio.checked = true;
      radio.addEventListener('change', update);
      return el('label', {}, radio, el('span', {}, g));
    }));
  form.append(el('div', { class: 'row' }, el('span', { class: 'row-head' }, 'Gender'), gender));

  for (const feat of SLIDERS) {
    const row = sliderRow(feat, numeric[feat], update);
    inputs[feat] = row;
    form.append(row.node);
  }

  const checks = el('div', { class: 'checks' });
  for (const feat of ['hypertension', 'heart_disease']) {
    const box = el('input', { type: 'checkbox', id: `in-${feat}` });
    box.addEventListener('change', update);
    inputs[feat] = box;
    checks.append(el('label', { for: `in-${feat}` }, box, LABELS[feat]));
  }
  form.append(checks);

  const select = el('select', { id: 'in-smoking' }, ...smoking.map((s) => new Option(s.label, s.value)));
  select.addEventListener('change', update);
  form.append(el('label', { class: 'field', for: 'in-smoking' }, el('span', {}, 'Smoking history'), select));
  form.addEventListener('submit', (e) => e.preventDefault());

  // Result area
  const cards = data.summary.best === 'Logistic Regression'
    ? ['Logistic Regression', 'Random Forest'] : ['Random Forest', 'Logistic Regression'];
  const readouts = {};
  const readoutWrap = el('div', { class: 'readouts' });
  for (const name of cards) {
    const value = el('p', { class: 'readout-value' }, '–');
    const band = el('p', { class: 'readout-band' }, '');
    const card = el('div', { class: 'readout', 'data-model': name },
      el('p', { class: 'readout-name' }, name), value, band);
    readouts[name] = { card, value, band };
    readoutWrap.append(card);
  }
  const markers = {
    'Random Forest': el('span', { class: 'marker rf' }, 'RF'),
    'Logistic Regression': el('span', { class: 'marker lr' }, 'LR'),
  };
  const track = el('div', { class: 'scale-track', 'aria-hidden': 'true' },
    el('span', { style: 'width:40%;background:var(--neg);opacity:.35' }),
    el('span', { style: 'width:30%;background:var(--mid);opacity:.45' }),
    el('span', { style: 'width:30%;background:var(--pos);opacity:.45' }));
  const scale = el('div', { class: 'scale', role: 'img', 'aria-label': 'Risk scale' },
    markers['Random Forest'], track, markers['Logistic Regression'],
    el('div', { class: 'scale-labels' }, el('span', {}, 'Low'), el('span', {}, 'Moderate'), el('span', {}, 'High')));
  const drivers = el('p', { class: 'drivers' });
  result.append(readoutWrap, scale, drivers);

  function patient() {
    const p = {
      gender: form.querySelector('input[name="gender"]:checked').value,
      smoking_history: select.value,
      hypertension: inputs.hypertension.checked ? 1 : 0,
      heart_disease: inputs.heart_disease.checked ? 1 : 0,
    };
    for (const feat of SLIDERS) p[feat] = clampInput(inputs[feat].number.value, numeric[feat]);
    return p;
  }

  function render() {
    const p = patient();
    const prior = data.summary.positive_rate;
    const probs = {
      'Random Forest': predictRF(data.model_rf, vectorize(data.model_rf.features, p)),
      'Logistic Regression': predictLR(data.model_lr, vectorize(data.model_lr.features, p)),
    };
    for (const [name, prob] of Object.entries(probs)) {
      const band = riskBand(prob, prior);
      const r = readouts[name];
      r.card.style.setProperty('--band', `var(${BAND_VAR[band]})`);
      r.value.textContent = formatProb(prob);
      r.band.textContent = `${band} estimated probability`;
      markers[name].style.left = `${Math.min(98, Math.max(2, prob * 100))}%`;
    }
    drivers.textContent = explain(p);
    const spoken = Object.entries(probs).map(([name, prob]) => `${name} ${formatProb(prob)}, ${riskBand(prob, prior).toLowerCase()}`).join('; ');
    scale.setAttribute('aria-label', `Risk scale: ${spoken}`);
    announce(spoken);
  }

  // Screen readers hear one short summary once the inputs settle, not every slider step.
  const summary = document.getElementById('predict-summary');
  let announceTimer;
  function announce(text) {
    clearTimeout(announceTimer);
    announceTimer = setTimeout(() => { if (summary) summary.textContent = text; }, 700);
  }

  function explain(p) {
    const names = explainLR(data.model_lr, vectorize(data.model_lr.features, p))
      .map((f) => SHORT[f] ?? LABELS[f] ?? f)
      .map((name) => (['HbA1c', 'BMI'].includes(name) ? name : name.toLowerCase()));
    if (!names.length) return 'Logistic Regression: nothing here pushes the estimate above an average patient.';
    return `Logistic Regression: the biggest push upward comes from ${names.join(' and ')}.`;
  }

  render();
}
