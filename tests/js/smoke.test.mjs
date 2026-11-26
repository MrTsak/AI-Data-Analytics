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
    assert.match(findings, /neither is suitable for medical screening/);
    assert.match(await page.textContent('#predict'), /trained on this dataset only/);
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

// Each group's histogram is a percentage of that group, so the two bars in a bin have different
// denominators and must overlap from zero. Stacked, a 20% and a 30% bar would end at 50%.
test('histogram bars overlap from zero instead of stacking', { skip: !chromium && 'playwright not installed', timeout: 60000 }, async () => {
  const server = await serve();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await page.goto(`http://localhost:${server.address().port}${PREFIX}`, { waitUntil: 'networkidle' });
    await page.waitForSelector('#explore-chart svg', { timeout: 10000 });
    const bars = await page.evaluate(async () => {
      const charts = await import('./js/charts.js');
      const c = charts.palette();
      // Bin 0 holds 20 of 100 records without diabetes (20%) and 30 of 100 with diabetes (30%).
      const plot = charts.histogram({ edges: [0, 1, 2], neg: [20, 80], pos: [30, 70] }, 'age', 600);
      const y = plot.scale('y');
      return [...plot.querySelectorAll('rect')]
        .filter((r) => [c.neg, c.pos].includes(r.getAttribute('fill')))
        .map((r) => ({
          group: r.getAttribute('fill') === c.neg ? 'No diabetes' : 'Diabetes',
          x: +r.getAttribute('x'),
          top: y.invert(+r.getAttribute('y')),
          bottom: y.invert(+r.getAttribute('y') + +r.getAttribute('height')),
        }));
    });
    const firstX = Math.min(...bars.map((b) => b.x));
    const bin0 = Object.fromEntries(bars.filter((b) => Math.abs(b.x - firstX) < 1).map((b) => [b.group, b]));
    assert.ok(bin0['No diabetes'] && bin0.Diabetes, `both groups drawn in the first bin: ${JSON.stringify(bars)}`);
    for (const [group, want] of [['No diabetes', 20], ['Diabetes', 30]]) {
      assert.ok(Math.abs(bin0[group].top - want) < 0.05, `${group} bar ends at ${bin0[group].top.toFixed(2)}%, expected ${want}%`);
      assert.ok(Math.abs(bin0[group].bottom) < 0.05, `${group} bar starts at ${bin0[group].bottom.toFixed(2)}%, expected 0%`);
    }
  } finally {
    await browser.close();
    server.close();
  }
});

// Tooltips show bin edges rounded for the variable and in the page's number format, not raw
// edges such as "77.336–80" with a decimal point on the Greek page.
test('histogram tooltips format bin edges in the page language', { skip: !chromium && 'playwright not installed', timeout: 60000 }, async () => {
  const server = await serve();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const tipFor = async (feat, edges) => {
      await page.evaluate(async ({ feat, edges, prefix }) => {
        const charts = await import(`${prefix}js/charts.js`);
        document.getElementById('tip-probe')?.remove();
        const plot = charts.histogram({ edges, neg: [20, 80], pos: [30, 70] }, feat, 600);
        plot.id = 'tip-probe';
        document.body.prepend(plot);
      }, { feat, edges, prefix: PREFIX });
      // Hover near the bottom of the first bin, where both groups' bars are drawn.
      const box = await page.locator('#tip-probe rect').first().boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height - 2);
      const tip = page.locator('#tip-probe [aria-label="tip"]');
      await tip.waitFor({ timeout: 5000 });
      return (await tip.textContent()).replace(/\s+/g, ' ').trim();
    };

    const port = server.address().port;
    await page.goto(`http://localhost:${port}${PREFIX}el/`, { waitUntil: 'networkidle' });
    await page.waitForSelector('#explore-chart svg', { timeout: 10000 });
    assert.match(await tipFor('HbA1c_level', [3.5, 3.683, 3.867]), /3,5–3,7: \d+,\d% της ομάδας/, 'Greek tooltip uses a decimal comma');
    assert.match(await tipFor('age', [77.336, 80, 82.664]), /77–80: \d+,\d% της ομάδας/, 'age edges in whole years');

    await page.goto(`http://localhost:${port}${PREFIX}`, { waitUntil: 'networkidle' });
    await page.waitForSelector('#explore-chart svg', { timeout: 10000 });
    assert.match(await tipFor('age', [77.336, 80, 82.664]), /77–80: \d+\.\d% of group/);
    assert.match(await tipFor('blood_glucose_level', [153.333, 160.667, 168]), /153–161: \d+\.\d% of group/);
    assert.match(await tipFor('bmi', [27.146, 30.002, 32.858]), /27\.1–30\.0: \d+\.\d% of group/);
  } finally {
    await browser.close();
    server.close();
  }
});

// Greek numbers use a decimal comma with a leading zero ("0,26"); English keeps its own format.
// Only the labels change: the bin edges, data and model outputs are untouched.
test('chart numbers follow each page language', { skip: !chromium && 'playwright not installed', timeout: 90000 }, async () => {
  const server = await serve();
  const browser = await chromium.launch();
  try {
    const decimalPoint = /(^|[^\d])(\.\d|\d\.\d{1,2}(?!\d))/; // 0.5, .26, 3.5 — but not Greek thousands like 17.530
    for (const lang of ['el', 'en']) {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
      await page.goto(`http://localhost:${server.address().port}${PREFIX}${lang === 'el' ? 'el/' : ''}`, { waitUntil: 'networkidle' });
      await page.waitForSelector('.readout-value', { timeout: 10000 });
      const shown = await page.evaluate(() => ({
        chartText: [...document.querySelectorAll('main svg text')].map((t) => t.textContent.trim()).filter((t) => /\d/.test(t)),
        corr: [...document.querySelectorAll('#corr-chart svg text')].map((t) => t.textContent.trim()).filter((t) => /^[\d.,]+$/.test(t)),
        roc: [...document.querySelectorAll('#roc-chart svg text')].map((t) => t.textContent.trim()),
        table: [...document.querySelectorAll('#metrics-table td')].map((t) => t.textContent),
        readouts: [...document.querySelectorAll('.readout-value')].map((t) => t.textContent),
        scale: document.querySelector('.scale').getAttribute('aria-label'),
      }));
      if (lang === 'el') {
        const offenders = [...shown.chartText, ...shown.table, ...shown.readouts, shown.scale].filter((t) => decimalPoint.test(t));
        assert.deepEqual(offenders, [], 'no decimal points on the Greek page');
        assert.ok(shown.corr.includes('0,26') && !shown.corr.some((v) => v.startsWith(',')), `correlation cells keep the leading zero: ${shown.corr.slice(0, 6)}`);
        assert.ok(shown.roc.includes('0,5'), 'ROC axis ticks use a comma');
        assert.match(shown.table.join(' '), /0,963/);
        assert.match(shown.scale, /\d,\d%/);
        // A tooltip, as a visitor sees it.
        await page.locator('#corr-chart').scrollIntoViewIfNeeded();
        const cell = await page.locator('#corr-chart svg rect').nth(1).boundingBox();
        await page.mouse.move(cell.x + cell.width / 2, cell.y + cell.height / 2);
        await page.waitForTimeout(200);
        const tip = await page.evaluate(() => [...document.querySelectorAll('#corr-chart svg text')].map((t) => t.textContent).find((t) => t.includes('r =')) || '');
        assert.match(tip, /r = 0,\d\d/, `correlation tooltip: ${tip}`);
      } else {
        assert.ok(shown.corr.includes('.26'), 'English correlation cells keep their compact form');
        assert.ok(shown.roc.includes('0.5'), 'English ROC ticks use a point');
        assert.match(shown.table.join(' '), /0\.963/);
      }
      // Every histogram bin keeps a distinct, ordered label that rounds its true edges.
      const bins = await page.evaluate(async () => {
        const charts = await import('../js/charts.js'.replace('../', document.documentElement.lang === 'el' ? '../' : './'));
        const res = await fetch(new URL('data/explore.json', new URL(document.documentElement.lang === 'el' ? '../' : './', location.href)));
        const { histograms } = await res.json();
        return Object.entries(histograms).map(([feat, h]) => ({
          feat, edges: h.edges, ranges: h.neg.map((_, i) => charts.binRange(feat, h.edges, i)),
        }));
      });
      const parse = (s) => Number(lang === 'el' ? s.replace(/\./g, '').replace(',', '.') : s.replace(/,/g, ''));
      for (const { feat, edges, ranges } of bins) {
        assert.equal(new Set(ranges).size, ranges.length, `${lang} ${feat}: every bin range is distinct`);
        ranges.forEach((r, i) => {
          const [lo, hi] = r.split('–');
          const decimals = (lo.split(lang === 'el' ? ',' : '.')[1] || '').length;
          const tolerance = 0.5 * 10 ** -decimals + 1e-9;
          assert.ok(parse(lo) < parse(hi), `${lang} ${feat} bin ${i}: ${r} increases`);
          assert.ok(Math.abs(parse(lo) - edges[i]) <= tolerance && Math.abs(parse(hi) - edges[i + 1]) <= tolerance, `${lang} ${feat} bin ${i}: ${r} rounds ${edges[i]}–${edges[i + 1]}`);
          if (lang === 'el') assert.doesNotMatch(r, decimalPoint, `${feat}: ${r} uses a decimal comma`);
        });
      }
      await page.close();
    }
  } finally {
    await browser.close();
    server.close();
  }
});

// The research sources are shown as background, apart from the dataset's own results, and the BMI
// claim cites NIDDK (the earlier ESC press-release link had been removed and showed "Page not found").
test('research sources are labelled as background and cite the checked NIDDK pages', { skip: !chromium && 'playwright not installed', timeout: 60000 }, async () => {
  const server = await serve();
  const browser = await chromium.launch();
  try {
    for (const [lang, lead, bmi] of [
      ['en', /Background from published sources, not results from this dataset/, /^Overweight and obesity are associated with a higher risk of type 2 diabetes \(source\)$/],
      ['el', /Στοιχεία από δημοσιευμένες πηγές, όχι αποτελέσματα από αυτό το σύνολο δεδομένων/, /^Το υπερβολικό βάρος και η παχυσαρκία συνδέονται με αυξημένο κίνδυνο εμφάνισης διαβήτη τύπου 2\. \(πηγή\)$/],
    ]) {
      const page = await browser.newPage();
      await page.goto(`http://localhost:${server.address().port}${PREFIX}${lang === 'el' ? 'el/' : ''}`, { waitUntil: 'networkidle' });
      await page.waitForSelector('#hypotheses li');
      assert.match(await page.textContent('.hypotheses-lead'), lead);
      const items = await page.$$eval('#hypotheses li', (lis) => lis.map((li) => ({ text: li.textContent.trim(), href: li.querySelector('a').href })));
      assert.equal(items.length, 4);
      assert.ok(items.every((i) => /^https:\/\//.test(i.href) && !i.href.includes('escardio.org')), JSON.stringify(items));
      // Every claim cites one of the two NIDDK pages it was checked against.
      const NIDDK = ['https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis', 'https://www.niddk.nih.gov/health-information/diabetes/overview/risk-factors-type-2-diabetes'];
      assert.deepEqual(items.map((i) => i.href), [NIDDK[0], NIDDK[1], NIDDK[1], NIDDK[0]], `${lang}: sources`);
      const expected = lang === 'el'
        ? [/^Τιμή HbA1c 5,7–6,4% υποδηλώνει προδιαβήτη/, /^Το υπερβολικό βάρος/, /^Ο κίνδυνος εμφάνισης διαβήτη τύπου 2 αυξάνεται με την ηλικία, ιδίως από τα 35/, /^Γλυκόζη αίματος νηστείας 100–125 mg\/dL υποδηλώνει προδιαβήτη· από 126 mg\/dL/]
        : [/^An HbA1c of 5\.7–6\.4% indicates prediabetes/, /^Overweight and obesity/, /^The risk of type 2 diabetes increases with age, especially from age 35/, /^A fasting blood glucose of 100–125 mg\/dL indicates prediabetes; 126 mg\/dL or above/];
      items.forEach((item, i) => assert.match(item.text, expected[i], `${lang} item ${i + 1}`));
      const bmiItem = items.find((i) => bmi.test(i.text));
      assert.ok(bmiItem, `${lang}: BMI background item: ${JSON.stringify(items.map((i) => i.text))}`);
      assert.equal(bmiItem.href, 'https://www.niddk.nih.gov/health-information/diabetes/overview/risk-factors-type-2-diabetes');
      await page.close();
    }
  } finally {
    await browser.close();
    server.close();
  }
});
