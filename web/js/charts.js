// Chart builders. Each returns a DOM node built with Observable Plot (loaded as a global).
import { t, featureLabel, modelName, num, pick } from './i18n.js';

const Plot = globalThis.Plot;

export const LABELS = {
  age: 'Age (years)',
  bmi: 'BMI',
  HbA1c_level: 'HbA1c (%)',
  blood_glucose_level: 'Blood glucose (mg/dL)',
  hypertension: 'Hypertension',
  heart_disease: 'Heart disease',
  gender: 'Gender',
  smoking_history: 'Smoking history',
  diabetes: 'Diabetes',
  gender_male: 'Male',
  smoking_never: 'Never smoked',
  smoking_former: 'Former smoker',
  smoking_current: 'Current smoker',
  smoking_not_current: 'Not currently smoking',
  smoking_ever: 'Has smoked',
};

export function palette() {
  const s = getComputedStyle(document.documentElement);
  const v = (n) => s.getPropertyValue(n).trim();
  return {
    ink: v('--ink'), soft: v('--ink-soft'), rule: v('--rule'), surface: v('--surface'),
    neg: v('--neg'), pos: v('--pos'), mid: v('--mid'), clusters: [v('--c0'), v('--c1'), v('--c2')],
  };
}

/**
 * Give a chart one accessible name. Plot labels its internal groups ("rect", "tip"...), which is
 * invalid on <g> and only noise for screen readers, so those labels are removed.
 */
export function describe(node, label) {
  const svg = node.tagName.toLowerCase() === 'svg' ? node : node.querySelector(':scope > svg');
  node.querySelectorAll('[aria-label]').forEach((el) => el.removeAttribute('aria-label'));
  if (svg) {
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', label);
  }
  return node;
}

const base = (c, width) => ({
  width,
  // Plot paints its tooltips with --plot-background; set it so tooltips follow the light/dark theme.
  style: `background: transparent; color: ${c.ink}; font-family: inherit; font-size: 13px; --plot-background: ${c.surface};`,
  marginLeft: 48,
  marginBottom: 46,
});

// Axis titles centred below the tick labels, so they never sit on top of the last tick.
const xAxis = (label, extra = {}) => ({ label, labelAnchor: 'center', labelOffset: 38, ...extra });

// Decimals shown for bin edges in histogram tooltips: enough that adjacent edges stay distinct
// (bins are about 2.7 years, 2.9 BMI, 0.18 HbA1c and 7.3 mg/dL wide), no more.
const EDGE_DIGITS = { age: 0, bmi: 1, HbA1c_level: 1, blood_glucose_level: 0 };

// Fallback for other features: one decimal place finer than the bin width.
const edgeDigits = (feat, edges) => {
  if (feat in EDGE_DIGITS) return EDGE_DIGITS[feat];
  const width = Math.abs(edges[1] - edges[0]);
  return width > 0 ? Math.max(0, Math.ceil(-Math.log10(width))) : 0;
};

/** Overlaid histograms as % of each group. */
export function histogram(h, feat, width) {
  const c = palette();
  const negTotal = h.neg.reduce((a, b) => a + b, 0);
  const posTotal = h.pos.reduce((a, b) => a + b, 0);
  const digits = edgeDigits(feat, h.edges);
  const rows = [];
  h.neg.forEach((n, i) => {
    rows.push({ x1: h.edges[i], x2: h.edges[i + 1], y: (n / negTotal) * 100, group: 'No diabetes' });
    rows.push({ x1: h.edges[i], x2: h.edges[i + 1], y: (h.pos[i] / posTotal) * 100, group: 'Diabetes' });
  });
  return Plot.plot({
    ...base(c, width), height: Math.min(380, width * 0.62),
    x: xAxis(featureLabel(feat, LABELS)),
    y: { label: t('% of group'), grid: true },
    color: { domain: ['No diabetes', 'Diabetes'], range: [c.neg, c.pos] },
    marks: [
      // Each group is a percentage of its own size, so the bars overlap from zero; stacking them
      // would add two percentages with different denominators. y1/y2 turn off Plot's implicit stack.
      Plot.rectY(rows, { x1: 'x1', x2: 'x2', y1: 0, y2: 'y', fill: 'group', insetLeft: 0.5, insetRight: 0.5,
        // Two cues besides colour: "No diabetes" is a solid bar, "Diabetes" an outlined, lighter one.
        fillOpacity: (d) => (d.group === 'Diabetes' ? 0.3 : 0.6),
        stroke: (d) => (d.group === 'Diabetes' ? c.pos : 'none'), strokeWidth: 1.5,
        tip: true,
        title: (d) => `${t(d.group)}\n${num(d.x1, digits)}–${num(d.x2, digits)}: ${num(d.y, 1)}% ${pick('of group', 'της ομάδας')}` }),
      Plot.ruleY([0], { stroke: c.soft }),
    ],
  });
}

/** Diabetes rate within each category. */
export function categoryRates(cat, key, width) {
  const c = palette();
  const rows = cat.labels.map((label, i) => {
    const total = cat.neg[i] + cat.pos[i];
    return { label: t(label), rate: (cat.pos[i] / total) * 100, total };
  });
  return Plot.plot({
    ...base(c, width), height: 60 + rows.length * 38, marginLeft: 170,
    x: xAxis(t('Records with diabetes (%)'), { grid: true }),
    y: { label: null, domain: rows.map((r) => r.label) },
    marks: [
      Plot.barX(rows, { y: 'label', x: 'rate', fill: c.pos, fillOpacity: 0.8,
        tip: true,
        title: (d) => `${d.label}\n${num(d.rate, 1)}% ${pick('of', 'από')} ${num(d.total)} ${pick('records', 'εγγραφές')}` }),
      Plot.text(rows, { y: 'label', x: 'rate', text: (d) => `${num(d.rate, 1)}%`, dx: 6, textAnchor: 'start', fill: c.ink }),
      Plot.ruleX([0], { stroke: c.soft }),
    ],
  });
}

// Short names without units for the correlation grid, which has little room on a phone.
const CORR_LABELS = {
  age: 'Age', hypertension: 'Hypertension', heart_disease: 'Heart disease', bmi: 'BMI',
  HbA1c_level: 'HbA1c', blood_glucose_level: 'Blood glucose', diabetes: 'Diabetes',
};

export function correlationHeatmap(corr, width) {
  const c = palette();
  const n = corr.labels.length;
  const name = (key) => featureLabel(key, CORR_LABELS);
  const rows = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      rows.push({ a: name(corr.labels[j]), b: name(corr.labels[i]), r: corr.matrix[i][j] });
    }
  }
  const names = corr.labels.map(name);
  const size = Math.min(width, 620);
  // Leave room for the longest row label (the Greek ones are longer), then fit the cells to what is left.
  const marginLeft = Math.ceil(16 + Math.max(...names.map((name) => name.length)) * 6.8);
  // On narrow screens the cells shrink, so drop the leading zero (".26") and use smaller text.
  const cell = (size - marginLeft - 20) / n;
  const fontSize = cell < 44 ? 10 : 11;
  const short = (r) => (r === 1 ? '1' : num(r, 2).replace(/^(-?)0(?=[.,])/, '$1'));
  return Plot.plot({
    ...base(c, size), height: size * 0.9, marginLeft, marginBottom: 110,
    x: { domain: names, label: null, tickRotate: -35 },
    y: { domain: names, label: null },
    color: { type: 'diverging', scheme: 'RdBu', domain: [-1, 1], reverse: true },
    marks: [
      Plot.cell(rows, { x: 'a', y: 'b', fill: 'r', inset: 1, tip: true,
        // Three short lines, so the tooltip fits on a phone screen.
        title: (d) => `${d.a}\n× ${d.b}\nr = ${num(d.r, 2)}` }),
      Plot.text(rows, { x: 'a', y: 'b', text: (d) => short(d.r),
        fill: (d) => (Math.abs(d.r) > 0.5 ? 'white' : 'black'), fontSize }),
    ],
  });
}

export const CLUSTER_SYMBOLS = ['circle', 'square', 'triangle'];

// Each dot is a set of records that share an HbA1c band, a BMI band and a cluster.
// Its size shows how many records it holds. Sets under 5 records are not published.
export function clusterScatter(grid, width) {
  const c = palette();
  return Plot.plot({
    ...base(c, width), height: Math.min(440, width * 0.7),
    x: xAxis('HbA1c (%)', { grid: true }),
    y: { label: 'BMI', grid: true },
    r: { range: [2, 12] },
    color: { domain: [0, 1, 2], range: c.clusters },
    // Each group also has its own shape, so the groups stay apart without colour.
    symbol: { domain: [0, 1, 2], range: CLUSTER_SYMBOLS },
    marks: [
      Plot.dot(grid, { x: 'HbA1c_level', y: 'bmi', fill: 'cluster', symbol: 'cluster', r: 'n', fillOpacity: 0.55 }),
    ],
  });
}

export function rocChart(metrics, names, width) {
  const c = palette();
  const colors = [c.ink, c.mid];
  const shown = names.map(modelName);
  const rows = names.flatMap((name) =>
    metrics[name].roc.fpr.map((fpr, i) => ({ name: modelName(name), fpr, tpr: metrics[name].roc.tpr[i] })));
  return Plot.plot({
    ...base(c, width), height: Math.min(380, width * 0.7),
    x: xAxis(t('False positive rate'), { domain: [0, 1] }),
    y: { label: t('True positive rate (recall)'), domain: [0, 1], grid: true },
    color: { domain: shown, range: colors },
    marks: [
      Plot.line([[0, 0], [1, 1]], { stroke: c.rule, strokeDasharray: '2 4' }),
      // Random Forest is a solid line, Logistic Regression a dashed one.
      ...shown.map((name, i) => Plot.line(rows.filter((d) => d.name === name),
        { x: 'fpr', y: 'tpr', stroke: colors[i], strokeWidth: 2.25, strokeDasharray: i ? '7 4' : null })),
    ],
  });
}

export function confusion(cm, title, width) {
  const c = palette();
  const labels = ['No diabetes', 'Diabetes'].map(t);
  const rows = [];
  cm.forEach((row, i) => row.forEach((v, j) =>
    rows.push({ actual: labels[i], predicted: labels[j], v, correct: i === j })));
  return Plot.plot({
    ...base(c, width), height: 260, marginLeft: 132, marginTop: 64, marginBottom: 8,
    title: modelName(title),
    x: { domain: labels, label: t('Model prediction'), axis: 'top', labelAnchor: 'center', labelOffset: 46 },
    y: { domain: labels, label: t('Actual class'), labelAnchor: 'center', labelOffset: 122 },
    marks: [
      Plot.cell(rows, { x: 'predicted', y: 'actual', fill: (d) => (d.correct ? c.neg : c.pos),
        fillOpacity: (d) => (d.correct ? 0.22 : 0.18), inset: 2 }),
      Plot.text(rows, { x: 'predicted', y: 'actual', text: (d) => num(d.v), fontSize: 18,
        fontWeight: 700, fill: c.ink }),
    ],
  });
}

export function importanceBars(importances, width) {
  const c = palette();
  const rows = Object.entries(importances)
    .map(([f, v]) => ({ f: featureLabel(f, LABELS), v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 8);
  return Plot.plot({
    ...base(c, width), height: 60 + rows.length * 30, marginLeft: 170,
    title: t('Which variables the Random Forest uses most'),
    x: xAxis(t('Feature importance'), { grid: true }),
    y: { label: null, domain: rows.map((r) => r.f) },
    marks: [
      Plot.barX(rows, { y: 'f', x: 'v', fill: c.ink, fillOpacity: 0.75 }),
      Plot.ruleX([0], { stroke: c.soft }),
    ],
  });
}
