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
