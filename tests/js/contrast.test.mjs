import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const css = readFileSync(new URL('../../web/styles.css', import.meta.url), 'utf8');
const lum = (hex) => {
  const c = hex.match(/\w\w/g).map((h) => parseInt(h, 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
};
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

test('band colours meet 4.5:1 on the predictor panel in both themes', () => {
  const blocks = [...css.matchAll(/--surface:\s*(#\w{6})[\s\S]*?(?=\}|$)/g)].map((m) => m[0]);
  assert.ok(blocks.length >= 2);
  for (const block of blocks) {
    const surface = block.match(/--surface:\s*(#\w{6})/)[1];
    for (const v of ['--neg', '--mid', '--pos']) {
      const m = block.match(new RegExp(`${v}:\\s*(#\\w{6})`));
      if (m) assert.ok(ratio(m[1], surface) >= 4.5, `${v} ${m[1]} on ${surface}: ${ratio(m[1], surface).toFixed(2)}`);
    }
  }
});
