/* IntelliReview — front end.
   One sheet of code on the left, the reviewer's margin on the right. No framework, no chart library. */
'use strict';

const API_BASE = location.protocol.startsWith('http') ? '' : 'http://127.0.0.1:8000';
const $ = id => document.getElementById(id);

// ─────────────────────────────────────────────────────────── state
const state = {
  lang: 'C',
  langPinned: false,        // the user chose a language by hand, so stop guessing it
  model: 'v3',
  data: null,               // last response
  code: '',                 // the code that response is about
  view: null,               // which model's verdict is on screen (compare mode)
  filter: 'ALL',
  reading: false,
  timers: [],
};

const SEV = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
const SEV_LABEL = { CRITICAL: 'Critical', HIGH: 'High', MEDIUM: 'Medium', LOW: 'Low' };
const LEVEL = { 'Clean': 0, 'Moderate Risk': 1, 'High Risk': 2 };
const LEVEL_TEXT = ['Clean', 'Moderate risk', 'High risk'];
const MODEL_NAME = {
  v3: 'V3 · path-sensitive forest', v4_dl: 'V4 · neural net, all features',
  ensemble: 'Boosted ensemble', dl: 'DL V1 · neural net, classic',
};
const MODEL_SHORT = { v3: 'V3', v4_dl: 'V4', ensemble: 'Ensemble', dl: 'DL V1' };

// ─────────────────────────────────────────────────────────── examples
const SAMPLES = {
  'C': `#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Finds max value in array */
int findMax(int *arr, int n) {
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

/* Bubble sort */
void bubbleSort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j+1]) {
                int tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }
}

void readName(char *buf) {
    gets(buf);
}

int main() {
    int *data = malloc(100 * sizeof(int));
    for (int i = 0; i < 100; i++) data[i] = rand() % 1000;
    bubbleSort(data, 100);
    printf("Max: %d\\n", findMax(data, 100));

    char name[16];
    readName(name);
    printf("Hello, %s\\n", name);
    return 0;
}`,
  'C++': `#include <iostream>
#include <vector>

class Buffer {
    int *cells;
public:
    Buffer(int n) { cells = new int[n]; }
    int at(int i) const { return cells[i]; }
};

int main() {
    Buffer *b = new Buffer(64);
    std::cout << b->at(3) << std::endl;
    return 0;
}`,
  'Java': `import java.sql.*;
import java.io.*;

public class Accounts {

    // Looks a user up by whatever name the caller typed.
    public static int findUser(Connection db) throws Exception {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String name = in.readLine();
        Statement st = db.createStatement();
        ResultSet rs = st.executeQuery("SELECT id FROM users WHERE name = '" + name + "'");
        return rs.next() ? rs.getInt(1) : -1;
    }

    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    public static void main(String[] args) throws Exception {
        System.out.println(fibonacci(10));
    }
}`,
};

// ─────────────────────────────────────────────────────────── elements
const editor = $('code-editor');
const gutter = $('gutter');
const listing = $('listing');
const sheet = $('sheet');
const analyzeBtn = $('analyze-btn');

// ─────────────────────────────────────────────────────────── editor
function refreshEditor() {
  const n = editor.value === '' ? 0 : editor.value.split('\n').length;
  let s = '';
  for (let i = 1; i <= Math.max(n, 1); i++) s += i + '\n';
  gutter.textContent = s;
  $('line-count').textContent = `${n} ${n === 1 ? 'line' : 'lines'}`;
  $('char-count').textContent = `${editor.value.length.toLocaleString()} characters`;
}
editor.addEventListener('input', () => {
  refreshEditor();
  if (state.data && !$('results-content').classList.contains('hidden')) {
    document.querySelector('#verdict .kicker').textContent = 'Verdict · the code has changed since';
  }
});
editor.addEventListener('scroll', () => { gutter.scrollTop = editor.scrollTop; });
editor.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); runAnalysis(); return; }
  const s = editor.selectionStart, v = editor.value;
  if (e.key === 'Tab') {
    e.preventDefault();
    document.execCommand ? document.execCommand('insertText', false, '    ') : (editor.setRangeText('    ', s, editor.selectionEnd, 'end'));
    refreshEditor();
  } else if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const lineStart = v.lastIndexOf('\n', s - 1) + 1;
    let indent = (v.slice(lineStart, s).match(/^[ \t]*/) || [''])[0];
    const opened = /[{(\[]$/.test(v.slice(lineStart, s).trimEnd());
    if (opened) indent += '    ';
    document.execCommand ? document.execCommand('insertText', false, '\n' + indent) : editor.setRangeText('\n' + indent, s, editor.selectionEnd, 'end');
    refreshEditor();
  }
});
editor.addEventListener('paste', () => setTimeout(() => { refreshEditor(); guessLanguage(); }, 0));
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && document.activeElement !== editor) { e.preventDefault(); runAnalysis(); }
});

// ─────────────────────────────────────────────────────────── language & model
function setLang(lang, byHand) {
  state.lang = lang;
  if (byHand) state.langPinned = true;
  document.querySelectorAll('#langs button').forEach(b => {
    const on = b.dataset.lang === lang;
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-pressed', on);
  });
}
document.querySelectorAll('#langs button').forEach(b => b.addEventListener('click', () => setLang(b.dataset.lang, true)));

function guessLanguage() {
  if (state.langPinned) return;
  const c = editor.value;
  if (/\b(public|private)\s+(static\s+)?(final\s+)?(class|interface|enum)\b|\bSystem\.out\b|\bimport\s+java\./.test(c)) return setLang('Java');
  if (/#\s*include\s*<(iostream|vector|string|map|memory|algorithm)>|\bstd::|\btemplate\s*<|\bclass\s+\w+\s*[:{]|\bnullptr\b/.test(c)) return setLang('C++');
  if (c.trim()) setLang('C');
}

$('model-select').addEventListener('change', e => { state.model = e.target.value; });

$('btn-clear').addEventListener('click', () => {
  leaveListing();
  editor.value = '';
  state.langPinned = false;
  refreshEditor();
  editor.focus();
});
$('btn-sample').addEventListener('click', () => {
  leaveListing();
  editor.value = SAMPLES[state.lang];
  refreshEditor();
  editor.scrollTop = 0;
});

// ─────────────────────────────────────────────────────────── files
function loadFile(file, thenReview) {
  if (!file) return;
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  const byExt = { java: 'Java', cpp: 'C++', cc: 'C++', cxx: 'C++', hpp: 'C++', c: 'C', h: 'C' }[ext];
  const reader = new FileReader();
  reader.onload = ev => {
    leaveListing();
    editor.value = String(ev.target.result || '');
    state.langPinned = false;
    if (byExt) setLang(byExt); else guessLanguage();
    refreshEditor();
    editor.scrollTop = 0;
    if (thenReview) setTimeout(runAnalysis, 120);
  };
  reader.onerror = () => showError(`The file “${file.name}” couldn’t be read.`);
  reader.readAsText(file);
}
$('file-input').addEventListener('change', e => { loadFile(e.target.files[0], true); e.target.value = ''; });
$('open-file').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file-input').click(); } });
['dragenter', 'dragover'].forEach(t => sheet.addEventListener(t, e => { e.preventDefault(); sheet.classList.add('is-drag'); }));
['dragleave', 'drop'].forEach(t => sheet.addEventListener(t, e => { e.preventDefault(); sheet.classList.remove('is-drag'); }));
sheet.addEventListener('drop', e => loadFile(e.dataTransfer.files[0], true));

// ─────────────────────────────────────────────────────────── screens in the margin
const screens = ['empty-state', 'loading-state', 'error-state', 'results-content'];
function show(id) { screens.forEach(s => $(s).classList.toggle('hidden', s !== id)); }

function showReading() {
  state.reading = true;
  sheet.classList.add('is-reading');
  analyzeBtn.disabled = true;
  $('analyze-btn-text').textContent = 'Reading…';
  show('loading-state');
  const items = [...document.querySelectorAll('#loader-steps li')];
  items.forEach((li, i) => li.className = i === 0 ? 'is-now' : '');
  state.timers.forEach(clearTimeout);
  state.timers = [700, 1500, 2400].map((ms, k) => setTimeout(() => {
    items.forEach((li, i) => li.className = i < k + 1 ? 'is-done' : i === k + 1 ? 'is-now' : '');
  }, ms));
}
function doneReading() {
  state.reading = false;
  state.timers.forEach(clearTimeout);
  sheet.classList.remove('is-reading');
  analyzeBtn.disabled = false;
  $('analyze-btn-text').textContent = state.data && !listing.classList.contains('hidden') ? 'Review again' : 'Review this code';
}
function showError(msg) {
  doneReading();
  $('error-message').textContent = msg;
  show('error-state');
}
$('retry-btn').addEventListener('click', runAnalysis);

// ─────────────────────────────────────────────────────────── the request
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (state.reading) return;
  if (!listing.classList.contains('hidden')) leaveListing();   // "Review again" works on the editor's text
  const code = editor.value;
  if (!code.trim()) { showError('There is nothing to review yet. Paste some code, open a file, or load an example.'); return; }

  showReading();
  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, language: state.lang, model_type: state.model }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const detail = Array.isArray(err.detail)
        ? err.detail.map(d => (d.msg || '').replace(/^Value error, /, '')).filter(Boolean).join('; ')
        : err.detail;
      throw new Error(detail || `The server answered with HTTP ${res.status}.`);
    }
    state.data = await res.json();
    state.code = code;
    await new Promise(r => setTimeout(r, 350));          // let the last reading step register
    render(state.data);
  } catch (err) {
    const offline = err instanceof TypeError;
    showError(offline ? 'The engine isn’t answering. Start it with run.bat, then try again.' : err.message);
  }
}

// ─────────────────────────────────────────────────────────── rendering
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const humanType = t => String(t || '').toLowerCase().replace(/_/g, ' ');
// how a finding reads inside a sentence
const PHRASE = {
  UNSAFE_FUNCTION: 'an unsafe library call', USE_AFTER_FREE: 'a use after free', DOUBLE_FREE: 'a double free',
  NULL_DEREF: 'a null dereference', BUFFER_OVERFLOW: 'a buffer overflow', FORMAT_STRING: 'a format-string bug',
  MEMORY_LEAK: 'a memory leak', RESOURCE_LEAK: 'a resource leak', UNINITIALIZED_VARIABLE: 'an uninitialized variable',
  SQL_INJECTION: 'SQL injection', COMMAND_INJECTION: 'command injection', INSECURE_DESERIALIZATION: 'unsafe deserialization',
  PATH_TRAVERSAL: 'path traversal', XSS: 'cross-site scripting', HARDCODED_CREDENTIAL: 'a hard-coded credential',
  WEAK_CRYPTO: 'weak cryptography', EMPTY_CATCH: 'a swallowed exception', UNCHECKED_ALLOC: 'an unchecked allocation',
  INVALID_FREE: 'an invalid free', INFINITE_LOOP: 'a possible infinite loop', HIGH_COMPLEXITY: 'a very complex function',
  DEEP_NESTING: 'deep nesting', RECURSION: 'recursion', DEAD_CODE: 'unreachable code',
};
const phrase = t => PHRASE[t] || humanType(t);
// message text from the engine: drop the "at line N" that the note already shows, and set `code` in mono
const prose = s => esc(String(s || '').replace(/\s+at line \d+(?=[:.,]| )/g, '')).replace(/`([^`]+)`/g, '<code>$1</code>');

function sortedIssues(issues) {
  return [...issues].sort((a, b) => SEV.indexOf(a.severity) - SEV.indexOf(b.severity) || (a.line || 1e9) - (b.line || 1e9));
}

function render(data) {
  doneReading();
  const risk = data.risk || {};
  const comps = risk.comparisons && Object.keys(risk.comparisons).length > 1 ? risk.comparisons : null;
  state.view = comps ? (comps.v3 ? 'v3' : Object.keys(comps)[0]) : state.model;

  const notes = sortedIssues(data.issues || []);
  state.notes = notes;
  state.filter = 'ALL';

  renderVerdict(comps ? comps[state.view] : risk, notes, data);
  renderCompare(comps);
  renderNotes(notes);
  renderLedger(data);
  renderBars(data.function_complexity || []);
  renderCalls(data);
  renderChanges((data.suggestions && data.suggestions.suggestions) || []);
  renderAnalysisNotes(data);
  enterListing(state.code, notes);

  show('results-content');
  $('verdict-label').focus({ preventScroll: true });
  $('analyze-btn-text').textContent = 'Review again';
}

function renderVerdict(r, notes, data) {
  const label = r.label || r.risk_label || 'Clean';
  const level = LEVEL[label] ?? 0;
  const score = Math.round(r.score ?? r.risk_score ?? 0);
  const v = $('verdict');
  v.dataset.level = level;
  v.querySelector('.kicker').textContent = 'Verdict';
  $('verdict-label').textContent = LEVEL_TEXT[level];
  $('verdict-score').textContent = score;
  requestAnimationFrame(() => { $('scale-mark').style.left = Math.max(0, Math.min(100, score)) + '%'; });
  $('verdict-model').textContent = MODEL_SHORT[state.view] || state.view;
  $('confidence-score').textContent = (r.confidence != null ? Math.round(r.confidence) : '—') + (r.confidence != null ? '%' : '');

  // one honest sentence about what the verdict rests on
  const heavy = notes.filter(n => n.severity === 'CRITICAL' || n.severity === 'HIGH').slice(0, 3);
  const mid = notes.filter(n => n.severity === 'MEDIUM').slice(0, 3);
  const pick = heavy.length ? heavy : mid;
  let why;
  const say = n => `${phrase(n.type)}${n.line ? ` on line ${n.line}` : ''}`;
  if (level === 2 && heavy.length) why = `It comes down to ${list(heavy.map(say))}.`;
  else if (level === 1 && pick.length) why = `Worth a look: ${list(pick.map(say))}.`;
  else if (level === 1) why = 'Nothing is clearly broken, but the shape of the code makes the models cautious.';
  else why = notes.length ? 'Only minor observations, none of which change the verdict.' : 'Nothing in here worried the analyzers.';
  // keep only the model remarks that tell the reader something (drop boilerplate and restatements of the findings)
  const extras = (r.explanations || []).filter(x => !(data.notes || []).includes(x) &&
    !/nominal variance|Raised to HIGH RISK|^Very high cyclomatic/i.test(x));
  const modelWord = extras.length ? ` <em>${esc(extras[extras.length - 1])}</em>` : '';
  $('verdict-why').innerHTML = esc(why) + modelWord;
}
const list = a => a.length < 2 ? (a[0] || '') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];

function renderCompare(comps) {
  const box = $('compare');
  if (!comps) { box.classList.add('hidden'); box.innerHTML = ''; return; }
  box.classList.remove('hidden');
  const order = ['v3', 'v4_dl', 'ensemble', 'dl'];
  const labels = new Set();
  let html = '';
  order.forEach(m => {
    const c = comps[m];
    if (!c) { html += `<button type="button" class="m-missing" disabled><span class="m-name">${MODEL_NAME[m]}</span><span class="m-verdict">not loaded</span><span class="m-score">–</span></button>`; return; }
    const lv = LEVEL[c.risk_label] ?? 0; labels.add(lv);
    html += `<button type="button" data-model="${m}" data-level="${lv}" class="${m === state.view ? 'is-on' : ''}"><span class="m-name">${MODEL_NAME[m]}</span><span class="m-verdict">${LEVEL_TEXT[lv]}</span><span class="m-score">${Math.round(c.risk_score)}</span></button>`;
  });
  html += `<p class="agree">${labels.size === 1 ? 'All of them agree.' : 'They disagree. The notes below are the same whichever you pick; only the verdict changes.'}</p>`;
  box.innerHTML = html;
  box.querySelectorAll('button[data-model]').forEach(b => b.addEventListener('click', () => {
    state.view = b.dataset.model;
    box.querySelectorAll('button').forEach(x => x.classList.toggle('is-on', x === b));
    renderVerdict(comps[state.view], state.notes, state.data);
  }));
}

function renderNotes(notes) {
  const counts = { ALL: notes.length };
  SEV.forEach(s => counts[s] = notes.filter(n => n.severity === s).length);
  $('tab-count-issues').textContent = notes.length;

  const f = $('filters');
  f.innerHTML = ['ALL', ...SEV].map(s =>
    `<button type="button" data-sev="${s}" class="${s === state.filter ? 'is-on' : ''}" ${counts[s] === 0 && s !== 'ALL' ? 'disabled' : ''}>${s === 'ALL' ? 'All' : SEV_LABEL[s]} ${counts[s]}</button>`).join('');
  f.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    state.filter = b.dataset.sev;
    f.querySelectorAll('button').forEach(x => x.classList.toggle('is-on', x === b));
    applyFilter();
  }));

  const ol = $('issues-list');
  ol.innerHTML = notes.map((n, i) => {
    const sev = n.severity.toLowerCase();
    const where = n.line ? `<button type="button" class="note__where" data-goto="${n.line}">line ${n.line}</button>` : `<span class="note__where" style="text-decoration:none;cursor:default">whole file</span>`;
    return `<li class="note sev-${sev}" data-i="${i}" data-sev="${n.severity}" data-line="${n.line || 0}" style="--i:${Math.min(i, 14)}">
      <span class="note__no">${i + 1}</span>
      <div class="note__top"><span class="tag sev-${sev}">${SEV_LABEL[n.severity] || esc(n.severity)}</span>${where}<span class="note__type">${esc(humanType(n.type))}</span></div>
      <p class="note__msg">${prose(n.message)}</p>
      ${n.suggestion ? `<p class="note__fix"><b>Change</b>${prose(n.suggestion)}</p>` : ''}
      ${n.fix ? `<div class="note__edit"><b>Suggested edit · line ${n.fix.line}</b>
        <pre class="diff"><del>- ${esc(n.fix.before.trim())}</del>
<ins>+ ${esc(n.fix.after.trim())}</ins></pre>
        ${n.fix.note ? `<p class="note__editnote">${esc(n.fix.note)}</p>` : ''}
        <button type="button" class="linkbtn" data-apply="${i}">Apply this edit and review again</button></div>` : ''}
    </li>`;
  }).join('');
  ol.querySelectorAll('[data-goto]').forEach(b => b.addEventListener('click', () => gotoLine(+b.dataset.goto)));
  ol.querySelectorAll('[data-apply]').forEach(b => b.addEventListener('click', () => applyFix(notes[+b.dataset.apply].fix)));
  ol.querySelectorAll('.note').forEach(li => {
    li.addEventListener('mouseenter', () => hotLine(+li.dataset.line, true));
    li.addEventListener('mouseleave', () => hotLine(+li.dataset.line, false));
  });
  applyFilter();
}
// replace one line of the reviewed code with the suggested edit, then review the result
function applyFix(fix) {
  const lines = state.code.split('\n');
  if (!fix || lines[fix.line - 1] !== fix.before) { showError('The code has changed since this review, so the edit can’t be applied safely. Review again first.'); return; }
  lines[fix.line - 1] = fix.after;
  leaveListing();
  editor.value = lines.join('\n');
  refreshEditor();
  runAnalysis();
}

function applyFilter() {
  let shown = 0;
  document.querySelectorAll('#issues-list .note').forEach(li => {
    const ok = state.filter === 'ALL' || li.dataset.sev === state.filter;
    li.classList.toggle('hidden', !ok);
    if (ok) shown++;
  });
  $('no-issues-msg').classList.toggle('hidden', shown > 0);
  $('no-issues-msg').textContent = state.notes.length === 0 ? 'Nothing to flag. That is not a guarantee, only what this analyzer can see.' : 'Nothing to flag in this category.';
}

function renderLedger(data) {
  const m = data.metrics || {};
  const rows = [
    ['Lines of code', m.lines_of_code],
    ['Functions', m.num_functions],
    ['Cyclomatic complexity', m.cyclomatic_complexity, 'highest of any function'],
    ['Estimated running time', m.time_complexity],
    ['Memory leaks', m.memory_leak_count, null, m.memory_leak_count > 0],
    ['Unsafe or injectable calls', m.unsafe_function_count, null, m.unsafe_function_count > 0],
    ['Recursive functions', m.recursion_count],
    ['Depth of analysis', data.analysis_mode === 'heuristic' ? 'heuristic' : 'full parse'],
    ...((data.suppressed || []).length ? [['Suppressed by comments', data.suppressed.length]] : []),
  ];
  $('ledger').innerHTML = rows.map(([k, v, , warn]) =>
    `<div class="${warn ? 'is-warn' : ''}"><dt>${esc(k)}</dt><span class="lead"></span><dd>${esc(v ?? '—')}</dd></div>`).join('');
}

function renderBars(fc) {
  const sec = $('sec-complexity');
  if (!fc.length) { sec.classList.add('hidden'); return; }
  sec.classList.remove('hidden');
  const max = Math.max(15, ...fc.map(f => f.cyclomatic_complexity));
  const rows = [...fc].sort((a, b) => b.cyclomatic_complexity - a.cyclomatic_complexity).slice(0, 14);
  $('complexity-bars').innerHTML = rows.map(f => {
    const cc = f.cyclomatic_complexity;
    const cls = cc > 15 ? 'is-hi' : cc > 10 ? 'is-mid' : '';
    return `<li><span class="b-name" title="${esc(f.function)}">${esc(f.function)}</span><span class="b-track"><span class="b-fill ${cls}" style="width:${(cc / max * 100).toFixed(1)}%"></span></span><span class="b-val">${cc}</span></li>`;
  }).join('') + (fc.length > rows.length ? `<li><span class="b-name" style="font-style:italic;font-family:var(--serif)">and ${fc.length - rows.length} more</span></li>` : '');
}

function renderCalls(data) {
  const sec = $('sec-structure');
  const fns = data.functions || [];
  if (!fns.length) { sec.classList.add('hidden'); return; }
  sec.classList.remove('hidden');
  const graph = {}; (data.cfgs || []).forEach(c => graph[c.func_name] = c.calls || []);
  const hasGraph = (data.cfgs || []).length > 0;
  $('functions-list').innerHTML = fns.slice(0, 40).map(f => {
    const calls = graph[f.name] || [];
    const selfCall = calls.includes(f.name);
    const others = calls.filter(c => c !== f.name);
    const to = !hasGraph ? '' :
      others.length ? `<span class="c-to">calls ${others.map(esc).join(', ')}${selfCall ? ' <span class="c-rec">· and itself ↻</span>' : ''}</span>`
        : selfCall ? '<span class="c-to c-rec">calls itself ↻</span>' : '<span class="c-to is-none">calls nothing else here</span>';
    return `<li><span class="c-fn">${esc(f.name)}<small>L${f.line || '?'}</small></span>${to}</li>`;
  }).join('') + (fns.length > 40 ? `<li><span class="c-to is-none">and ${fns.length - 40} more</span></li>` : '');
}

function renderChanges(sugs) {
  $('tab-count-suggestions').textContent = sugs.length;
  const sec = $('sec-changes');
  if (!sugs.length) { sec.classList.add('hidden'); return; }
  sec.classList.remove('hidden');
  $('suggestions-list').innerHTML = sugs.map(s => `
    <div class="change">
      <h4>${prose(s.message)}</h4>
      <p>${prose(s.suggestion)}${s.line ? ` <span class="note__type">· line ${s.line}</span>` : ''}</p>
      ${s.example ? `<pre>${esc(s.example)}</pre>` : ''}
    </div>`).join('');
}

function renderAnalysisNotes(data) {
  const box = $('analysis-notes');
  const notes = data.notes || [];
  box.hidden = notes.length === 0;
  box.innerHTML = notes.map(n => `<p>${esc(n)}</p>`).join('');
}

// ─────────────────────────────────────────────────────────── the annotated listing
const KEYWORDS = new Set(('auto break case char const continue default do double else enum extern float for goto if inline int long register ' +
  'restrict return short signed sizeof static struct switch typedef union unsigned void volatile while class public private protected new delete ' +
  'template typename namespace using try catch throw this nullptr bool true false null final abstract implements extends interface import package ' +
  'instanceof synchronized throws boolean byte super finally var').split(' '));

function highlight(code) {
  let inBlock = false;
  return code.split('\n').map(raw => {
    let out = '', i = 0;
    const put = (cls, text) => { out += cls ? `<span class="${cls}">${esc(text)}</span>` : esc(text); };
    if (!inBlock && /^\s*#/.test(raw)) { put('tk-p', raw); return out; }
    while (i < raw.length) {
      if (inBlock) {
        const end = raw.indexOf('*/', i);
        if (end === -1) { put('tk-c', raw.slice(i)); i = raw.length; } else { put('tk-c', raw.slice(i, end + 2)); i = end + 2; inBlock = false; }
        continue;
      }
      const rest = raw.slice(i);
      let m;
      if (rest.startsWith('//')) { put('tk-c', rest); break; }
      if (rest.startsWith('/*')) {
        const end = raw.indexOf('*/', i + 2);
        if (end === -1) { put('tk-c', rest); inBlock = true; i = raw.length; }
        else { put('tk-c', raw.slice(i, end + 2)); i = end + 2; }
        continue;
      }
      if ((m = /^"(?:\\.|[^"\\])*"?/.exec(rest)) || (m = /^'(?:\\.|[^'\\])*'?/.exec(rest))) { put('tk-s', m[0]); i += m[0].length; continue; }
      if ((m = /^\d[\w.]*/.exec(rest))) { put('tk-n', m[0]); i += m[0].length; continue; }
      if ((m = /^[A-Za-z_]\w*/.exec(rest))) { put(KEYWORDS.has(m[0]) ? 'tk-k' : '', m[0]); i += m[0].length; continue; }
      put('', raw[i]); i++;
    }
    return out || '&nbsp;';
  });
}

function enterListing(code, notes) {
  const byLine = {};
  notes.forEach((n, i) => { if (n.line > 0) (byLine[n.line] = byLine[n.line] || []).push({ n, no: i + 1 }); });
  const lines = highlight(code);
  listing.innerHTML = lines.map((h, idx) => {
    const ln = idx + 1, hits = byLine[ln];
    const worst = hits ? hits.map(x => x.n.severity).sort((a, b) => SEV.indexOf(a) - SEV.indexOf(b))[0].toLowerCase() : '';
    const pins = hits ? hits.map(x => `<button type="button" class="pin sev-${x.n.severity.toLowerCase()}" data-note="${x.no}" title="Note ${x.no}: ${esc(SEV_LABEL[x.n.severity])}" aria-label="Note ${x.no}, line ${ln}"><span>${x.no}</span></button>`).join('') : '';
    return `<div class="ln${hits ? ' is-flag sev-' + worst : ''}" id="ln-${ln}" data-ln="${ln}"><span class="ln__no">${ln}</span><span class="ln__pin">${pins}</span><span class="ln__code">${h}</span></div>`;
  }).join('');
  listing.querySelectorAll('.pin').forEach(p => p.addEventListener('click', () => gotoNote(+p.dataset.note)));

  editor.classList.add('hidden');
  gutter.classList.add('hidden');
  document.querySelector('.sheet__body').classList.add('is-listing');
  listing.classList.remove('hidden');
  $('btn-edit').classList.remove('hidden');
  listing.scrollTop = 0;
  const first = notes.find(n => n.line > 0);
  if (first) setTimeout(() => scrollListingTo(first.line, false), 250);
}
function leaveListing() {
  listing.classList.add('hidden');
  editor.classList.remove('hidden');
  gutter.classList.remove('hidden');
  document.querySelector('.sheet__body').classList.remove('is-listing');
  $('btn-edit').classList.add('hidden');
  $('analyze-btn-text').textContent = 'Review this code';
}
$('btn-edit').addEventListener('click', () => { leaveListing(); editor.focus(); });

function scrollListingTo(line, pulse = true) {
  const el = $('ln-' + line);
  if (!el) return;
  const top = el.offsetTop - listing.clientHeight / 2 + el.clientHeight / 2;
  listing.scrollTo({ top: Math.max(0, top), behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  if (pulse) { el.classList.remove('is-pulse'); void el.offsetWidth; el.classList.add('is-pulse'); }
}
function gotoLine(line) { scrollListingTo(line); }
function gotoNote(no) {
  const li = document.querySelector(`#issues-list .note[data-i="${no - 1}"]`);
  if (!li) return;
  if (li.classList.contains('hidden')) { state.filter = 'ALL'; renderNotes(state.notes); }
  const target = document.querySelector(`#issues-list .note[data-i="${no - 1}"]`);
  target.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
  target.classList.add('is-hot'); setTimeout(() => target.classList.remove('is-hot'), 1200);
}
function hotLine(line, on) { const el = line && $('ln-' + line); if (el) el.classList.toggle('is-hot', on); }

// ─────────────────────────────────────────────────────────── save / print
$('print-btn').addEventListener('click', () => window.print());
$('download-report-btn').addEventListener('click', () => {
  const r = state.data; if (!r) return;
  const risk = r.risk || {}, m = r.metrics || {};
  const comps = risk.comparisons && Object.keys(risk.comparisons).length > 1 ? risk.comparisons : null;
  const view = comps ? comps[state.view] : risk;
  const level = LEVEL[view.label || view.risk_label] ?? 0;
  const rule = '='.repeat(64);
  let t = `${rule}\nINTELLIREVIEW — REVIEW\n${rule}\n`;
  t += `Language  : ${r.language || state.lang}\nDate      : ${new Date().toLocaleString()}\nAnalysis  : ${r.analysis_mode === 'heuristic' ? 'heuristic (not fully parsed)' : 'full parse'}\n\n`;
  t += `VERDICT   : ${LEVEL_TEXT[level]}  (${Math.round(view.score ?? view.risk_score ?? 0)}/100)\n`;
  if (comps) Object.entries(comps).forEach(([k, c]) => { t += `  ${(MODEL_SHORT[k] || k).padEnd(9)}: ${LEVEL_TEXT[LEVEL[c.risk_label] ?? 0]} (${Math.round(c.risk_score)})\n`; });
  t += `\nMEASUREMENTS\n${'-'.repeat(12)}\n`;
  [['Lines of code', m.lines_of_code], ['Functions', m.num_functions], ['Cyclomatic complexity', m.cyclomatic_complexity],
   ['Estimated running time', m.time_complexity], ['Memory leaks', m.memory_leak_count], ['Unsafe calls', m.unsafe_function_count],
   ['Recursive functions', m.recursion_count]].forEach(([k, v]) => { t += `${k.padEnd(26)} ${v}\n`; });
  t += `\nNOTES (${state.notes.length})\n${'-'.repeat(8)}\n`;
  if (!state.notes.length) t += 'Nothing to flag.\n';
  state.notes.forEach((n, i) => {
    t += `\n[${i + 1}] ${n.severity}${n.line ? `, line ${n.line}` : ''}  (${humanType(n.type)})\n    ${n.message}\n`;
    if (n.suggestion) t += `    Change: ${n.suggestion}\n`;
  });
  const sugs = (r.suggestions && r.suggestions.suggestions) || [];
  if (sugs.length) {
    t += `\nCHANGES WORTH MAKING\n${'-'.repeat(20)}\n`;
    sugs.forEach((s, i) => { t += `\n${i + 1}. ${s.message}\n   ${s.suggestion}\n`; if (s.example) s.example.split('\n').forEach(l => t += `     ${l}\n`); });
  }
  (r.notes || []).forEach(n => t += `\nNote: ${n}\n`);
  t += `\n${rule}\nThe analyzer reports what it can see. It is not a guarantee that code is safe.\n`;
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([t], { type: 'text/plain;charset=utf-8' })), download: `intellireview-${Date.now()}.txt` });
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
});

// ─────────────────────────────────────────────────────────── engine status
async function pingEngine() {
  const el = $('engine'), txt = $('engine-text');
  try {
    const r = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
    if (!r.ok) throw new Error();
    el.className = 'engine is-up'; txt.textContent = 'Engine online';
  } catch { el.className = 'engine is-down'; txt.textContent = 'Engine offline'; }
}
pingEngine();
setInterval(pingEngine, 20000);

// ─────────────────────────────────────────────────────────── start
refreshEditor();
