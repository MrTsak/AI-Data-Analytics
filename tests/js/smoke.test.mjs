// Browser smoke test: serves web/ under /projects/diabetes/ (as on the portfolio site) and
// checks every section renders and the predictor reacts. Skips if Playwright isn't installed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright'));
} catch {
  try { ({ chromium } = require('/opt/node-tools/node_modules/playwright')); } catch { chromium = null; }
}
// In CI the browser tests must run: fail loudly rather than skip.
if (!chromium && process.env.REQUIRE_BROWSER) throw new Error('REQUIRE_BROWSER is set but Playwright is not installed');

const WEB_ROOT = process.env.WEB_ROOT || fileURLToPath(new URL('../../web/', import.meta.url));
const PREFIX = '/projects/diabetes/';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };

function serve() {
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    if (!url.pathname.startsWith(PREFIX)) { res.writeHead(404).end(); return; }
    let rel = normalize(url.pathname.slice(PREFIX.length)).replace(/^(\.\.[/\\])+/, '');
    if (rel === '' || rel === '.') rel = 'index.html';
    else if (rel.endsWith('/')) rel = join(rel, 'index.html');
    try {
      const body = await readFile(join(WEB_ROOT, rel));
      res.writeHead(200, { 'content-type': TYPES[extname(rel)] || 'application/octet-stream' }).end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  return new Promise((resolve) => server.listen(0, () => resolve(server)));
}

test('dashboard renders under a sub-path and the predictor responds', { skip: !chromium && 'playwright not installed', timeout: 60000 }, async () => {
  const server = await serve();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('response', (r) => { if (r.url().includes(PREFIX) && r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
    await page.goto(`http://localhost:${server.address().port}${PREFIX}`, { waitUntil: 'networkidle' });
    await page.waitForSelector('.readout-value', { timeout: 10000 });

    for (const id of ['explore-chart', 'corr-chart', 'cluster-chart', 'roc-chart', 'confusion-charts', 'importance-chart']) {
      assert.ok(await page.$(`#${id} svg`), `#${id} has a chart`);
    }
    assert.ok(await page.$('#metrics-table tbody tr'), 'metrics table filled');
    assert.ok((await page.$$('#findings li')).length >= 4, 'findings listed');

    const rf = page.locator('.readout[data-model="Random Forest"] .readout-value');
    const before = await rf.textContent();
    await page.fill('#in-HbA1c_level', '9');
    await page.fill('#in-blood_glucose_level', '250');
    assert.notEqual(await rf.textContent(), before, 'RF probability changed');

    await page.fill('#in-bmi', '');
    assert.match(await rf.textContent(), /\d/, 'empty input still gives a number');

    // Review fixes: honest copy, no smoking in the explanation, step-aligned inputs.
    const findings = await page.textContent('#findings');
    assert.match(findings, /neither is suitable for screening/);
    assert.match(await page.textContent('#predict'), /describe this dataset only/);
    await page.selectOption('#in-smoking', 'never');
    assert.doesNotMatch(await page.textContent('.drivers'), /smok/i);
    assert.match(await page.textContent('.drivers'), /^Logistic Regression/);
    await page.fill('#in-age', '');
    await page.focus('#in-age-range');
    await page.keyboard.press('ArrowRight');
    assert.match(await page.inputValue('#in-age'), /^\d+$/, 'age stays a whole number');

    // Screen readers: a short delayed summary, not the whole result on every step.
    assert.equal(await page.getAttribute('#predict-result', 'aria-live'), null);
    await page.waitForFunction(() => /Random Forest [<>]?\d/.test(document.querySelector('#predict-summary')?.textContent || ''));
    assert.match(await page.getAttribute('.scale', 'aria-label'), /Random Forest .*%.*Logistic Regression .*%/);

    // Phones keep the section links.
    await page.setViewportSize({ width: 390, height: 844 });
    const nav = await page.locator('.topbar nav a').first().boundingBox();
    assert.ok(nav && nav.width > 0, 'nav visible on phones');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), 390);

    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    server.close();
  }
});

test('Greek page renders the same charts in Greek', { skip: !chromium && 'playwright not installed', timeout: 60000 }, async () => {
  const server = await serve();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('response', (r) => { if (r.url().includes(PREFIX) && r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
    await page.goto(`http://localhost:${server.address().port}${PREFIX}el/`, { waitUntil: 'networkidle' });
    await page.waitForSelector('.readout-value', { timeout: 10000 });

    assert.equal(await page.getAttribute('html', 'lang'), 'el');
    for (const id of ['explore-chart', 'corr-chart', 'cluster-chart', 'roc-chart', 'confusion-charts', 'importance-chart']) {
      assert.ok(await page.$(`#${id} svg`), `#${id} has a chart`);
    }
    assert.match(await page.textContent('[data-bind="hero-lede"]'), /96\.128 εγγραφές/, 'record count uses Greek separators');
    assert.match(await page.textContent('#cluster-list li strong'), /^Ομάδα 1 — /);
    assert.match(await page.textContent('#cluster-list li span'), /εγγραφές · Μέση ηλικία: 49 έτη · Μέσο BMI: 28,9/);
    assert.match(await page.textContent('#metrics-table'), /Δείκτης αξιολόγησης/);
    assert.match(await page.textContent('#findings'), /Το Random Forest πέτυχε/);
    assert.match(await page.textContent('#findings'), /ιατρικό προσυμπτωματικό έλεγχο/);
    assert.equal(await page.textContent('.theme-toggle'), 'Σκούρο θέμα');

    // The predictor responds in Greek too.
    assert.match(await page.textContent('.readout[data-model="Random Forest"] .readout-band'), /εκτίμηση του μοντέλου/);
    assert.match(await page.textContent('.drivers'), /^Λογιστική παλινδρόμηση:/);
    const rf = page.locator('.readout[data-model="Random Forest"] .readout-value');
    const before = await rf.textContent();
    await page.fill('#in-HbA1c_level', '9');
    assert.notEqual(await rf.textContent(), before, 'Greek RF probability changed');
    assert.match(await rf.textContent(), /%/);

    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    server.close();
  }
});
