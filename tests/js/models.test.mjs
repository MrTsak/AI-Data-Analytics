import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { predictLR, predictRF, vectorize, riskBand, clampInput } from '../../web/js/models.js';

const load = (n) => JSON.parse(readFileSync(new URL(`../../web/data/${n}`, import.meta.url)));
const lr = load('model_lr.json');
const rf = load('model_rf.json');
const parity = load('parity.json');

test('LR matches Python on all parity cases', () => {
  for (const c of parity) {
    const p = predictLR(lr, vectorize(lr.features, c.input));
    assert.ok(Math.abs(p - c.lr) < 1e-6, `lr ${p} vs ${c.lr}`);
  }
});

test('RF matches Python on all parity cases', () => {
  for (const c of parity) {
    const p = predictRF(rf, vectorize(rf.features, c.input));
    assert.ok(Math.abs(p - c.rf) < 1e-6, `rf ${p} vs ${c.rf}`);
  }
});

test('riskBand boundaries', () => {
  assert.equal(riskBand(0.39), 'Low');
  assert.equal(riskBand(0.4), 'Moderate');
  assert.equal(riskBand(0.69), 'Moderate');
  assert.equal(riskBand(0.7), 'High');
});

test('clampInput handles empty, NaN and out-of-range', () => {
  const r = { min: 10, max: 95, mean: 27.3 };
  assert.equal(clampInput('', r), 27.3);
  assert.equal(clampInput('abc', r), 27.3);
  assert.equal(clampInput(999, r), 95);
  assert.equal(clampInput('-5', r), 10);
  assert.equal(clampInput('30.5', r), 30.5);
});

test('vectorize encodes raw gender and smoking values', () => {
  const x = vectorize(lr.features, {
    gender: 'Male', smoking_history: 'former', age: 50, hypertension: 1, heart_disease: 0,
    bmi: 30, HbA1c_level: 6.5, blood_glucose_level: 150,
  });
  const at = (f) => x[lr.features.indexOf(f)];
  assert.equal(at('gender_male'), 1);
  assert.equal(at('smoking_former'), 1);
  for (const f of lr.features.filter((f) => f.startsWith('smoking_') && f !== 'smoking_former')) {
    assert.equal(at(f), 0);
  }
  assert.equal(at('bmi'), 30);
});
