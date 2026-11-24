// Browser-side model runtime. Pure functions, no DOM: tested with `node --test tests/js/`.

const SMOKING_COLUMNS = {
  never: 'smoking_never',
  former: 'smoking_former',
  current: 'smoking_current',
  'not current': 'smoking_not_current',
  ever: 'smoking_ever',
};

/** Turn a patient object into a feature vector in `features` order.
 *  Accepts raw form values ({gender, smoking_history, ...}) or already-encoded features. */
export function vectorize(features, input) {
  const encoded = { ...input };
  if ('gender' in input) encoded.gender_male = input.gender === 'Male' ? 1 : 0;
  if ('smoking_history' in input) {
    for (const col of Object.values(SMOKING_COLUMNS)) encoded[col] = 0;
    const col = SMOKING_COLUMNS[input.smoking_history];
    if (col) encoded[col] = 1;
  }
  return features.map((f) => Number(encoded[f] ?? 0));
}

/** Logistic regression on standardised features. */
export function predictLR(model, x) {
  let z = model.intercept;
  for (let i = 0; i < x.length; i++) {
    z += model.coef[i] * ((x[i] - model.mean[i]) / model.scale[i]);
  }
  return 1 / (1 + Math.exp(-z));
}

/** Random forest: average of each tree's leaf probability.
 *  sklearn casts inputs to float32 before comparing, so we do the same with Math.fround. */
export function predictRF(model, x) {
  const xf = x.map(Math.fround);
  let total = 0;
  for (const tree of model.trees) {
    let n = 0;
    while (tree.f[n] !== -1) {
      n = xf[tree.f[n]] <= tree.t[n] ? tree.l[n] : tree.r[n];
    }
    total += tree.v[n];
  }
  return total / model.trees.length;
}

export function riskBand(p) {
  if (p >= 0.7) return 'High';
  if (p >= 0.4) return 'Moderate';
  return 'Low';
}

/** Parse a typed value; empty or invalid falls back to the mean, out-of-range is clamped. */
export function clampInput(value, { min, max, mean }) {
  const n = typeof value === 'number' ? value : parseFloat(String(value).trim());
  if (!Number.isFinite(n)) return mean;
  return Math.min(max, Math.max(min, n));
}

/** Features that push the logistic-regression score up most versus an average patient.
 *  Smoking categories are left out: their coefficients mostly reflect who has no smoking
 *  record (largely children) rather than an effect of smoking, so naming them misleads. */
export function explainLR(model, x, { threshold = 0.15, limit = 2 } = {}) {
  return model.features
    .map((f, i) => ({ f, c: model.coef[i] * ((x[i] - model.mean[i]) / model.scale[i]) }))
    .filter((d) => !d.f.startsWith('smoking_') && d.c > threshold)
    .sort((a, b) => b.c - a.c)
    .slice(0, limit)
    .map((d) => d.f);
}

/** Slider bounds and start value aligned to the step, so arrow keys land on round values. */
export function stepAlign({ p01, p99, mean, step }) {
  const decimals = step < 1 ? 1 : 0;
  const round = (v) => Number(v.toFixed(decimals));
  const lo = round(Math.floor(p01 / step + 1e-9) * step);
  const hi = round(Math.ceil(p99 / step - 1e-9) * step);
  const start = round(Math.min(hi, Math.max(lo, Math.round(mean / step) * step)));
  return { lo, hi, start };
}
