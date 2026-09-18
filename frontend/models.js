/* Fills models.html from /api/models — the page never hard-codes an accuracy figure. */
'use strict';
const API = location.protocol.startsWith('http') ? '' : 'http://127.0.0.1:8000';
const q = s => document.querySelector(s);
const pct = x => x == null ? '—' : Math.round(x * 100) + '%';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

(async function () {
  let payload;
  try {
    const r = await fetch(`${API}/api/models`, { cache: 'no-store' });
    if (!r.ok) throw new Error();
    payload = await r.json();
  } catch {
    q('#model-table tbody').innerHTML = '<tr><td colspan="4">The engine isn’t running, so the live numbers can’t be loaded. Start it with run.bat and reload.</td></tr>';
    q('#results-table tbody').innerHTML = '<tr><td colspan="7">—</td></tr>';
    return;
  }

  q('#model-table tbody').innerHTML = payload.models.map(m => `
    <tr><td><b>${esc(m.name)}</b><span class="sm">${esc(m.short)}</span></td>
    <td>${esc(m.kind)}</td><td class="num">${m.features}</td>
    <td>${esc(m.inputs)}<span class="sm">${esc(m.note)}</span></td></tr>`).join('');

  const rep = payload.report;
  const body = q('#results-table tbody');
  if (!rep) {
    body.innerHTML = '<tr><td colspan="7">No evaluation is on file yet. Run <code>python ml/retrain_all.py</code> in <code>backend/</code>.</td></tr>';
    return;
  }
  const R = rep.results || {};
  const j = k => (R[k] && R[k].juliet) || null;
  const rows = [];
  const add = (label, sub, key, cls, useRaw) => {
    const x = j(key); if (!x) return;
    rows.push(`<tr class="${cls || ''}"><td>${label}<span class="sm">${sub || ''}</span></td>
      <td class="num">${pct(x.acc)}</td><td class="num">${useRaw ? '—' : pct(x.acc_gated)}</td>
      <td class="num">${(useRaw ? x.macro_f1 : x.macro_f1_gated).toFixed(2)}</td>
      <td class="num">${useRaw ? '—' : pct(x.false_alarm_clean)}</td><td class="num">${useRaw ? '—' : pct(x.false_high_clean)}</td>
      <td class="num">${useRaw ? '—' : pct(x.high_recall)}</td></tr>`);
  };
  add('File size only', 'the shortcut the old models had learned', 'size_only', 'baseline', true);
  add('The rules, no models', 'what the analysis finds by itself', 'rule_engine_only');
  [['v3', 'V3 · path-sensitive forest'], ['ensemble', 'Boosted ensemble'], ['dl', 'DL V1 · neural net, classic'], ['v4_dl', 'V4 · neural net, all features']]
    .forEach(([k, name]) => add(name, '', k));
  body.innerHTML = rows.join('');

  const n = (j('rule_engine_only') || {}).n;
  q('#results-caption').textContent =
    `${n ? n.toLocaleString() : '—'} labelled functions from the NIST Juliet suites, ${rep.folds}-fold, whole vulnerability classes held out. ` +
    `“Accuracy alone” is the model’s raw answer; “as used” includes the evidence rule above. The two neural networks were tested on a single held-out fold.`;

  const size = j('size_only');
  q('#quote').textContent = size
    ? `An earlier version of these models reported 100% accuracy. It was measuring nothing: the training data was built so that bigger files meant riskier files, and the models had learned file size. The first row above is that shortcut on real data. Beating it is the bar.`
    : '';

  const cb = rep.class_balance || [];
  const total = rep.n_samples || cb.reduce((a, b) => a + b, 0);
  q('#data-text').innerHTML =
    `The main source is the NIST <b>Juliet Test Suite v1.3</b> for C/C++ and for Java: thousands of test cases, each with a deliberately flawed function and a fixed twin. ` +
    `They are used one function at a time, with comments stripped so the labels can’t leak. To stop file size from predicting the answer, the training set also ` +
    `glues random correct functions, and at most one flawed one, into larger files, and adds generated programs for the Moderate cases. ` +
    `In total ${total.toLocaleString()} labelled files: ${cb[0] ? cb[0].toLocaleString() : '—'} clean, ${cb[1] ? cb[1].toLocaleString() : '—'} moderate, ${cb[2] ? cb[2].toLocaleString() : '—'} high risk.`;
})();
