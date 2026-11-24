// The Greek page (web/el/index.html) is a hand-written translation of web/index.html. These checks
// fail when the two drift apart, so a change to one page is not forgotten on the other.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const web = fileURLToPath(new URL('../../web/', import.meta.url));
const en = readFileSync(`${web}index.html`, 'utf8');
const el = readFileSync(`${web}el/index.html`, 'utf8');

const all = (html, re) => [...html.matchAll(re)].map((m) => m[1]);
const meta = (html, prop) => (html.match(new RegExp(`property="${prop}" content="([^"]*)"`)) || [])[1];

test('both pages have the same element ids in the same order', () => {
  assert.deepEqual(all(el, /\sid="([^"]+)"/g), all(en, /\sid="([^"]+)"/g));
});

test('both pages have the same sections, headings and list items', () => {
  const shape = (html) => ({
    sections: (html.match(/<section\b/g) || []).length,
    headings: all(html, /<(h[1-3])\b/g),
    items: (html.match(/<li\b/g) || []).length,
    paragraphs: (html.match(/<p\b/g) || []).length,
    links: (html.match(/<a\b/g) || []).length,
  });
  assert.deepEqual(shape(el), shape(en));
});

test('languages, canonical URLs and alternates point at each other', () => {
  assert.match(en, /<html lang="en">/);
  assert.match(el, /<html lang="el">/);
  const canonical = (html) => (html.match(/rel="canonical" href="([^"]+)"/) || [])[1];
  assert.equal(canonical(en), 'https://tsakioan.com/projects/diabetes/');
  assert.equal(canonical(el), 'https://tsakioan.com/projects/diabetes/el/');
  for (const html of [en, el]) {
    assert.match(html, /hreflang="en" href="https:\/\/tsakioan\.com\/projects\/diabetes\/"/);
    assert.match(html, /hreflang="el" href="https:\/\/tsakioan\.com\/projects\/diabetes\/el\/"/);
  }
  assert.equal(meta(en, 'og:url'), canonical(en));
  assert.equal(meta(el, 'og:url'), canonical(el));
  assert.equal(meta(en, 'og:locale'), 'en_US');
  assert.equal(meta(el, 'og:locale'), 'el_GR');
});

test('link-preview images and the favicon exist', () => {
  for (const html of [en, el]) {
    const image = meta(html, 'og:image');
    assert.ok(image, 'og:image set');
    const file = image.replace('https://tsakioan.com/projects/diabetes/', '').replace(/\?.*$/, '');
    assert.ok(existsSync(`${web}${file}`), `${file} exists in web/`);
  }
  assert.match(en, /rel="icon" href="favicon\.svg"/);
  assert.match(el, /rel="icon" href="\.\.\/favicon\.svg"/);
  assert.ok(existsSync(`${web}favicon.svg`));
});
