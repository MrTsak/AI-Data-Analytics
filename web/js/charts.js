// Chart builders. Each returns a DOM node built with Observable Plot (loaded as a global).
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

const base = (c, width) => ({
  width,
  style: { background: 'transparent', color: c.ink, fontFamily: 'inherit', fontSize: '13px' },
  marginLeft: 48,
});

/** Overlaid histograms as % of each group. */
export function histogram(h, feat, width) {
  const c = palette();
  const negTotal = h.neg.reduce((a, b) => a + b, 0);
  const posTotal = h.pos.reduce((a, b) => a + b, 0);
  const rows = [];
  h.neg.forEach((n, i) => {
    rows.push({ x1: h.edges[i], x2: h.edges[i + 1], y: (n / negTotal) * 100, group: 'No diabetes' });
    rows.push({ x1: h.edges[i], x2: h.edges[i + 1], y: (h.pos[i] / posTotal) * 100, group: 'Diabetes' });
  });
  return Plot.plot({
    ...base(c, width), height: Math.min(380, width * 0.62),
    x: { label: LABELS[feat] },
    y: { label: '% of group', grid: true },
    color: { domain: ['No diabetes', 'Diabetes'], range: [c.neg, c.pos] },
    marks: [
      Plot.rectY(rows, { x1: 'x1', x2: 'x2', y: 'y', fill: 'group', fillOpacity: 0.55, inset: 0.5,
        tip: true, title: (d) => `${d.group}\n${d.x1}–${d.x2}: ${d.y.toFixed(1)}% of group` }),
      Plot.ruleY([0], { stroke: c.soft }),
    ],
  });
}

/** Diabetes rate within each category. */
export function categoryRates(cat, key, width) {
  const c = palette();
  const rows = cat.labels.map((label, i) => {
    const total = cat.neg[i] + cat.pos[i];
    return { label, rate: (cat.pos[i] / total) * 100, total };
  });
  return Plot.plot({
    ...base(c, width), height: 60 + rows.length * 38, marginLeft: 170,
    x: { label: 'Diabetes rate (%)', grid: true },
    y: { label: null, domain: rows.map((r) => r.label) },
    marks: [
      Plot.barX(rows, { y: 'label', x: 'rate', fill: c.pos, fillOpacity: 0.8,
        tip: true, title: (d) => `${d.label}: ${d.rate.toFixed(1)}% of ${d.total.toLocaleString()} patients` }),
      Plot.text(rows, { y: 'label', x: 'rate', text: (d) => `${d.rate.toFixed(1)}%`, dx: 6, textAnchor: 'start', fill: c.ink }),
      Plot.ruleX([0], { stroke: c.soft }),
    ],
  });
}

export function correlationHeatmap(corr, width) {
  const c = palette();
  const n = corr.labels.length;
  const rows = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      rows.push({ a: LABELS[corr.labels[j]], b: LABELS[corr.labels[i]], r: corr.matrix[i][j] });
    }
  }
  const names = corr.labels.map((l) => LABELS[l]);
  const size = Math.min(width, 620);
  return Plot.plot({
    ...base(c, size), height: size * 0.9, marginLeft: 150, marginBottom: 110,
    x: { domain: names, label: null, tickRotate: -35 },
    y: { domain: names, label: null },
    color: { type: 'diverging', scheme: 'RdBu', domain: [-1, 1], reverse: true },
    marks: [
      Plot.cell(rows, { x: 'a', y: 'b', fill: 'r', inset: 1, tip: true,
        title: (d) => `${d.a} × ${d.b}: r = ${d.r.toFixed(2)}` }),
      Plot.text(rows, { x: 'a', y: 'b', text: (d) => d.r.toFixed(2),
        fill: (d) => (Math.abs(d.r) > 0.5 ? 'white' : 'black'), fontSize: 11 }),
    ],
  });
}

// Each dot is a group of patients who share an HbA1c band, a BMI band and a cluster.
// Its size shows how many patients it holds. Groups under 5 people are not published.
export function clusterScatter(grid, width) {
  const c = palette();
  return Plot.plot({
    ...base(c, width), height: Math.min(440, width * 0.7),
    x: { label: 'HbA1c (%)', grid: true },
    y: { label: 'BMI', grid: true },
    r: { range: [2, 12] },
    color: { domain: [0, 1, 2], range: c.clusters },
    marks: [
      Plot.dot(grid, { x: 'HbA1c_level', y: 'bmi', fill: 'cluster', r: 'n', fillOpacity: 0.55 }),
    ],
  });
}

export function rocChart(metrics, names, width) {
  const c = palette();
  const colors = [c.ink, c.mid];
  const rows = names.flatMap((name) =>
    metrics[name].roc.fpr.map((fpr, i) => ({ name, fpr, tpr: metrics[name].roc.tpr[i] })));
  return Plot.plot({
    ...base(c, width), height: Math.min(380, width * 0.7),
    x: { label: 'False positive rate', domain: [0, 1] },
    y: { label: 'True positive rate (recall)', domain: [0, 1], grid: true },
    color: { domain: names, range: colors, legend: true },
    marks: [
      Plot.line([[0, 0], [1, 1]], { stroke: c.rule, strokeDasharray: '4 4' }),
      Plot.line(rows, { x: 'fpr', y: 'tpr', stroke: 'name', strokeWidth: 2.25 }),
    ],
  });
}

export function confusion(cm, title, width) {
  const c = palette();
  const labels = ['No diabetes', 'Diabetes'];
  const rows = [];
  cm.forEach((row, i) => row.forEach((v, j) =>
    rows.push({ actual: labels[i], predicted: labels[j], v, correct: i === j })));
  return Plot.plot({
    ...base(c, width), height: 230, marginLeft: 100, marginTop: 34,
    title,
    x: { domain: labels, label: 'Predicted', axis: 'top' },
    y: { domain: labels, label: 'Actual' },
    marks: [
      Plot.cell(rows, { x: 'predicted', y: 'actual', fill: (d) => (d.correct ? c.neg : c.pos),
        fillOpacity: (d) => (d.correct ? 0.22 : 0.18), inset: 2 }),
      Plot.text(rows, { x: 'predicted', y: 'actual', text: (d) => d.v.toLocaleString(), fontSize: 18,
        fontWeight: 700, fill: c.ink }),
    ],
  });
}

export function importanceBars(importances, width) {
  const c = palette();
  const rows = Object.entries(importances)
    .map(([f, v]) => ({ f: LABELS[f] ?? f, v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 8);
  return Plot.plot({
    ...base(c, width), height: 60 + rows.length * 30, marginLeft: 170,
    title: 'What the Random Forest relies on',
    x: { label: 'Feature importance', grid: true },
    y: { label: null, domain: rows.map((r) => r.f) },
    marks: [
      Plot.barX(rows, { y: 'f', x: 'v', fill: c.ink, fillOpacity: 0.75 }),
      Plot.ruleX([0], { stroke: c.soft }),
    ],
  });
}
