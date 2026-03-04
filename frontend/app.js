/**
 * IntelliReview — Frontend Application Logic
 * Handles: code editor, API calls, risk gauge, Chart.js complexity chart, Mermaid graph
 */

// Dynamically determine API base URL for better cross-platform compatibility
const API_BASE = (() => {
  // If we're being served from the backend, use relative paths
  if (typeof window !== 'undefined' && window.location.hostname) {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = window.location.port ? ':' + window.location.port : '';
    return `${protocol}//${hostname}${port}`;
  }
  return 'http://localhost:8000';
})();

// ── State ──────────────────────────────────────────────────────────────
let currentLanguage = 'C';
let complexityChart = null;
let currentResults = null;
let currentIssueFilter = 'ALL';

// ── Sample Code ────────────────────────────────────────────────────────
const SAMPLES = {
  C: `#include <stdio.h>
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

/* Bubble sort — O(n^2) */
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

/* Reads user name — UNSAFE */
void readName(char *buf) {
    gets(buf);  /* dangerous! */
}

int main() {
    int *data = malloc(100 * sizeof(int));
    /* NOTE: no free() — memory leak */
    for (int i = 0; i < 100; i++) data[i] = rand() % 1000;
    bubbleSort(data, 100);
    printf("Max: %d\\n", findMax(data, 100));

    char name[16];
    readName(name);
    printf("Hello, %s\\n", name);
    return 0;
}`,

  Java: `import java.util.ArrayList;
import java.util.Scanner;

public class DataProcessor {

    // Fibonacci — recursive
    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    // Find all pairs summing to target — O(n^2)
    public static ArrayList<int[]> findPairs(int[] arr, int target) {
        ArrayList<int[]> result = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[i] + arr[j] == target) {
                    if (arr[i] > 0) {
                        if (arr[j] > 0) {
                            result.add(new int[]{arr[i], arr[j]});
                        }
                    }
                }
            }
        }
        return result;
    }

    // Build string in loop — inefficient
    public static String buildReport(String[] items) {
        String report = "";
        for (String item : items) {
            report = report.concat(item + "\\n");
        }
        return report;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] data = {3, 5, 1, 8, 2, 9, 4, 7, 6};
        System.out.println("Pairs: " + findPairs(data, 10));
        System.out.println("Fib(10): " + fibonacci(10));
        System.out.println(buildReport(new String[]{"a","b","c"}));
    }
}`,
};

// ── DOM References ─────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const codeEditor = $('code-editor');
const lineNumbers = $('line-numbers');
const analyzeBtn = $('analyze-btn');
const fileInput = $('file-input');
const dropzone = $('dropzone');

// State panels
const emptyState = $('empty-state');
const loadingState = $('loading-state');
const resultsContent = $('results-content');
const errorState = $('error-state');

// ── Language Toggle ────────────────────────────────────────────────────
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentLanguage = btn.dataset.lang;
    $('editor-lang-indicator').textContent = currentLanguage;
  });
});

// ── Editor Line Numbers ────────────────────────────────────────────────
function updateLineNumbers() {
  const lines = codeEditor.value.split('\n');
  lineNumbers.textContent = lines.map((_, i) => i + 1).join('\n');
  $('line-count').textContent = `${lines.length} lines`;
  $('char-count').textContent = `${codeEditor.value.length} chars`;
}

codeEditor.addEventListener('input', updateLineNumbers);
codeEditor.addEventListener('scroll', () => {
  lineNumbers.scrollTop = codeEditor.scrollTop;
});
codeEditor.addEventListener('keydown', e => {
  const s = codeEditor.selectionStart;
  const val = codeEditor.value;

  // TAB: Insert 4 spaces
  if (e.key === 'Tab') {
    e.preventDefault();
    codeEditor.value = val.substring(0, s) + '    ' + val.substring(codeEditor.selectionEnd);
    codeEditor.selectionStart = codeEditor.selectionEnd = s + 4;
    updateLineNumbers();
  }

  // Auto-indent on Enter
  if (e.key === 'Enter') {
    e.preventDefault();
    // Find indentation of current line
    const currentLineStart = val.lastIndexOf('\n', s - 1) + 1;
    const currentLine = val.substring(currentLineStart, s);
    const indentMatch = currentLine.match(/^\s*/);
    let indent = indentMatch ? indentMatch[0] : '';

    // If the previous character was an opening brace/bracket, add extra indent
    if (val[s - 1] === '{' || val[s - 1] === '(' || val[s - 1] === '[') {
      indent += '    ';
    }

    codeEditor.value = val.substring(0, s) + '\n' + indent + val.substring(codeEditor.selectionEnd);
    codeEditor.selectionStart = codeEditor.selectionEnd = s + 1 + indent.length;

    // Auto-close brace behavior (optional UX)
    if (val[s - 1] === '{' && val.substring(s).trim().startsWith('}')) {
      // if they press enter between {}
      const pre = codeEditor.value.substring(0, codeEditor.selectionStart);
      const post = codeEditor.value.substring(codeEditor.selectionEnd);
      const unindent = indent.substring(0, indent.length - 4);
      codeEditor.value = pre + '\n' + unindent + post;
      codeEditor.selectionStart = codeEditor.selectionEnd = pre.length;
    }

    updateLineNumbers();
  }
});
updateLineNumbers();

// ── Clear & Sample ─────────────────────────────────────────────────────
$('btn-clear').addEventListener('click', () => {
  codeEditor.value = '';
  updateLineNumbers();
});
$('btn-sample').addEventListener('click', () => {
  codeEditor.value = SAMPLES[currentLanguage];
  updateLineNumbers();
});

// ── File Upload ────────────────────────────────────────────────────────
function loadFileIntoEditor(file, autoAnalyze = false) {
  if (!file) return;
  const ext = file.name.split('.').pop().toLowerCase();

  // Auto-detect language from extension
  if (ext === 'java') {
    currentLanguage = 'Java';
    document.querySelectorAll('.lang-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.lang === 'Java')
    );
    $('editor-lang-indicator').textContent = 'Java';
  } else {
    currentLanguage = 'C';
    document.querySelectorAll('.lang-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.lang === 'C')
    );
    $('editor-lang-indicator').textContent = 'C';
  }

  const reader = new FileReader();
  reader.onload = ev => {
    codeEditor.value = ev.target.result;
    updateLineNumbers();

    // Show which file was loaded
    const dropLabel = dropzone.querySelector('p');
    if (dropLabel) dropLabel.textContent = `✅ ${file.name} loaded`;

    // Auto-analyze after file is fully read
    if (autoAnalyze) {
      setTimeout(runAnalysis, 200);
    }
  };
  reader.onerror = () => showError(`Failed to read file: ${file.name}`);
  reader.readAsText(file);

  // IMPORTANT: Reset input value so same file can be re-selected
  fileInput.value = '';
}

fileInput.addEventListener('change', e => {
  loadFileIntoEditor(e.target.files[0], true);   // auto-analyze = true
});

// Drag & drop
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  loadFileIntoEditor(e.dataTransfer.files[0], true);  // auto-analyze = true
});
dropzone.addEventListener('click', () => fileInput.click());

// ── Analyze ────────────────────────────────────────────────────────────
console.log('🎬 IntelliReview initialized');
console.log(`Button found: ${analyzeBtn ? 'YES' : 'NO'}`);
if (analyzeBtn) {
  analyzeBtn.addEventListener('click', runAnalysis);
  console.log('✅ Analyze button listener attached');
} else {
  console.error('❌ Analyze button (id="analyze-btn") not found!');
}
$('retry-btn').addEventListener('click', runAnalysis);

async function runAnalysis() {
  console.log('🔘 Analyze button clicked');

  const code = codeEditor.value.trim();
  console.log(`📝 Code captured: ${code.length} characters`);

  if (!code) {
    console.warn('⚠️ No code provided');
    showError('Please paste or upload some code first.');
    return;
  }

  const modelType = $('model-select').value;
  console.log(`🤖 Model selected: ${modelType}`);
  console.log(`💬 Language: ${currentLanguage}`);
  console.log(`🌍 API Base: ${API_BASE}`);

  showLoading();

  // Animate loader steps
  const steps = document.querySelectorAll('.loader-step');
  const delays = [0, 800, 1600, 2400];
  steps.forEach((s, i) => {
    s.classList.remove('active', 'done');
    setTimeout(() => {
      steps.forEach(x => x.classList.remove('active'));
      s.classList.add('active');
      if (i > 0) steps[i - 1].classList.add('done');
    }, delays[i]);
  });

  try {
    console.log(`📤 Sending POST request to ${API_BASE}/api/analyze`);
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, language: currentLanguage, model_type: modelType }),
    });

    console.log(`📥 Response status: ${response.status} ${response.statusText}`);

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Server error' }));
      console.error('❌ Server error:', err);
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ Analysis completed:', data);
    currentResults = data;
    await new Promise(r => setTimeout(r, 400)); // brief pause for final step
    steps[steps.length - 1].classList.add('done');
    await new Promise(r => setTimeout(r, 300));
    renderResults(data);
  } catch (err) {
    console.error('💥 Analysis error:', err.message);
    showError(err.message || 'Could not connect to the backend. Make sure the server is running on port 8000.');
  }
}

// ── Download Report ────────────────────────────────────────────────────
$('download-report-btn').addEventListener('click', () => {
  if (!currentResults) return;

  const r = currentResults;
  const metrics = r.metrics;
  const issues = r.issues || [];
  const suggestions = (r.suggestions && r.suggestions.suggestions) || [];
  const modelLabel = $('model-select').options[$('model-select').selectedIndex].text
    .replace(/[^\u0000-\u007E]/g, '').trim(); // strip emojis

  const SEP = '================================================================';
  const sec = (title) => `\n${title}\n${'─'.repeat(title.length)}\n`;

  let report = `${SEP}\nINTELLIREVIEW — AI CODE ANALYSIS REPORT\n${SEP}\n`;
  report += `Language : ${r.language || currentLanguage}\n`;
  report += `Model    : ${modelLabel}\n`;
  report += `Generated: ${new Date().toLocaleString()}\n`;

  const hasComparisons = r.risk.comparisons && Object.keys(r.risk.comparisons).length > 1;
  const MODEL_NAMES = { ensemble: 'Ensemble (LightGBM + XGBoost + ET)', dl: 'DL V1 (19-feat)', v3: 'V3 Path-Sensitive RF', v4_dl: 'V4 Dedup DL (36-feat)' };

  if (hasComparisons) {
    // ── All-Models Report: full section per model ──────────────────────
    report += `\n${SEP}\nSECTION 1 — RISK SCORES (ALL MODELS)\n${SEP}\n`;
    for (const [mType, comp] of Object.entries(r.risk.comparisons)) {
      const name = MODEL_NAMES[mType] || mType.toUpperCase();
      report += `\n[ ${name} ]\n`;
      report += `  Label : ${comp.risk_label}\n`;
      report += `  Score : ${comp.risk_score}/100\n`;
      report += `  Confidence : ${comp.confidence ?? '-'}%\n`;
      const p = comp.probabilities || {};
      report += `  Probabilities: Clean ${p['Clean'] ?? 0}%  |  Moderate ${p['Moderate Risk'] ?? 0}%  |  High ${p['High Risk'] ?? 0}%\n`;
      if (comp.explanations && comp.explanations.length) {
        report += `  Explanation: ${comp.explanations[comp.explanations.length - 1]}\n`;
      }
    }
  } else {
    // ── Single Model Report ────────────────────────────────────────────
    report += sec('RISK SCORE');
    report += `Label : ${r.risk.label}\n`;
    report += `Score : ${r.risk.score}/100\n`;
    const p = r.risk.probabilities || {};
    report += `Probabilities: Clean ${p['Clean'] ?? 0}%  |  Moderate ${p['Moderate Risk'] ?? 0}%  |  High ${p['High Risk'] ?? 0}%\n`;
  }

  // ── Metrics (always) ────────────────────────────────────────────────
  report += `\n${SEP}\nSECTION 2 — CODE METRICS\n${SEP}\n`;
  report += `Lines of Code          : ${metrics.lines_of_code}\n`;
  report += `Number of Functions    : ${metrics.num_functions}\n`;
  report += `Cyclomatic Complexity  : ${metrics.cyclomatic_complexity}\n`;
  report += `Time Complexity        : ${metrics.time_complexity}\n`;
  report += `Memory Leaks Detected  : ${metrics.memory_leak_count}\n`;
  report += `Unsafe C Functions     : ${metrics.unsafe_function_count}\n`;
  report += `Recursion Count        : ${metrics.recursion_count}\n`;
  report += `\nVulnerability Categories:\n`;
  report += `  Memory & Pointers    : ${r.risk.categories?.memory || 0}\n`;
  report += `  Data Flow (SSA)      : ${r.risk.categories?.data_flow || 0}\n`;
  report += `  Architecture/Smells  : ${r.risk.categories?.architecture || 0}\n`;
  report += `  Recursion Hazards    : ${r.risk.categories?.recursion || 0}\n`;

  // ── Issues (always) ─────────────────────────────────────────────────
  report += `\n${SEP}\nSECTION 3 — ISSUES DETECTED (${issues.length})\n${SEP}\n`;
  if (issues.length === 0) {
    report += 'No issues detected.\n';
  } else {
    issues.forEach((iss, idx) => {
      report += `\n[${idx + 1}] SEVERITY : ${iss.severity}\n`;
      if (iss.line) report += `    Line    : ${iss.line}\n`;
      if (iss.type) report += `    Type    : ${iss.type}\n`;
      report += `    Message : ${iss.message}\n`;
      if (iss.suggestion) report += `    Fix     : ${iss.suggestion}\n`;
    });
  }

  // ── Suggestions (always) ────────────────────────────────────────────
  report += `\n${SEP}\nSECTION 4 — IMPROVEMENT SUGGESTIONS (${suggestions.length})\n${SEP}\n`;
  if (suggestions.length === 0) {
    report += 'No suggestions.\n';
  } else {
    suggestions.forEach((s, idx) => {
      report += `\n[${idx + 1}] ${s.category} (${s.severity})\n`;
      report += `    Message    : ${s.message}\n`;
      report += `    Suggestion : ${s.suggestion}\n`;
      if (s.example) {
        report += `    Example:\n`;
        s.example.split('\n').forEach(l => { report += `      ${l}\n`; });
      }
    });
  }

  // ── Trigger download ────────────────────────────────────────────────
  const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `IntelliReview_Report_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// ── State Management ───────────────────────────────────────────────────
function showLoading() {
  emptyState.classList.add('hidden');
  errorState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  loadingState.classList.remove('hidden');
  analyzeBtn.classList.add('loading');
  $('analyze-btn-text') && ($('analyze-btn-text').textContent = 'Analyzing...');
}

function showError(msg) {
  emptyState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  errorState.classList.remove('hidden');
  $('error-message').textContent = msg;
  analyzeBtn.classList.remove('loading');
  $('analyze-btn-text') && ($('analyze-btn-text').textContent = 'Analyze Code');
}

// ── Render Results ─────────────────────────────────────────────────────
function renderResults(data) {
  try {
    loadingState.classList.add('hidden');
    analyzeBtn.classList.remove('loading');
    $('analyze-btn-text') && ($('analyze-btn-text').textContent = 'Analyze Code');
    resultsContent.classList.remove('hidden');

    if (!data) { showError('Invalid response from server'); return; }

    const { metrics = {}, risk = {}, issues = [], suggestions = {}, function_complexity = [], functions = [], cfgs = [] } = data;

    // ── ML Panel (gauge + probabilities + confidence) ─────────────────
    renderMLPanel(risk);

    // ── Model Switcher (Compare All mode) ─────────────────────────────
    const switcher = $('model-switcher');
    const compCard = $('comparison-card');
    const hasComparisons = risk.comparisons && Object.keys(risk.comparisons).length > 1;

    if (hasComparisons) {
      switcher.classList.remove('hidden');
      compCard && (compCard.style.display = 'none'); // hide old mini-card
      renderModelSwitcher(risk.comparisons, risk);
    } else {
      switcher.classList.add('hidden');
      compCard && (compCard.style.display = 'none');
    }

    // ── Metrics ───────────────────────────────────────────────────────
    $('m-loc').textContent = metrics.lines_of_code;
    $('m-cc').textContent = metrics.cyclomatic_complexity;
    $('m-time').textContent = metrics.time_complexity;
    $('m-fn').textContent = metrics.num_functions;
    $('m-leak').textContent = metrics.memory_leak_count;
    $('m-unsafe').textContent = metrics.unsafe_function_count;

    if (metrics.memory_leak_count > 0) $('metric-leak').classList.add('warn');
    if (metrics.unsafe_function_count > 0) $('metric-unsafe').classList.add('warn');

    // ── Issues ────────────────────────────────────────────────────────
    try { renderIssues(issues || []); $('tab-count-issues').textContent = (issues || []).length; }
    catch (e) { console.error('Issues render error:', e); }

    // ── Suggestions ───────────────────────────────────────────────────
    try {
      const sugs = (suggestions && suggestions.suggestions) || [];
      renderSuggestions(sugs);
      $('tab-count-suggestions').textContent = sugs.length;
    } catch (e) { console.error('Suggestions render error:', e); }

    // ── Complexity Chart ──────────────────────────────────────────────
    try { renderComplexityChart(function_complexity || []); }
    catch (e) { console.error('Complexity render error:', e); }

    // ── Functions & CFG ───────────────────────────────────────────────
    try { renderFunctions(functions || [], function_complexity || [], cfgs || []); }
    catch (e) { console.error('Functions render error:', e); }

  } catch (e) {
    console.error('Error rendering results:', e);
    showError(`Rendering error: ${e.message}`);
  }
}

// ── ML Panel: Gauge + Proba + Confidence ──────────────────────────────────
function renderMLPanel(riskObj) {
  renderGauge(riskObj.score || riskObj.risk_score || 0, riskObj.label || riskObj.risk_label || 'Unknown');

  const proba = riskObj.probabilities || {};
  $('proba-clean').textContent = (proba['Clean'] ?? 0) + '%';
  $('proba-moderate').textContent = (proba['Moderate Risk'] ?? 0) + '%';
  $('proba-high').textContent = (proba['High Risk'] ?? 0) + '%';

  if ($('confidence-score')) {
    const conf = riskObj.confidence || 0;
    $('confidence-score').textContent = conf + '%';
    const exList = riskObj.explanations || [];
    $('confidence-explain').textContent = exList.length > 0
      ? exList[exList.length - 1]
      : 'Prediction bounds aligned successfully.';

    if (conf === 100 && (riskObj.label || riskObj.risk_label) === 'High Risk') {
      $('confidence-explain').style.color = 'var(--clr-high)';
    } else if (conf < 70) {
      $('confidence-explain').style.color = 'var(--clr-moderate)';
    } else {
      $('confidence-explain').style.color = 'var(--clr-text-3)';
    }
  }
}

// ── Model Switcher: wire up 4 buttons in Compare All mode ─────────────────
const MODEL_META = {
  v3: { icon: '🔬', name: 'V3 Path-Sensitive' },
  dl: { icon: '🧠', name: 'DL V1' },
  ensemble: { icon: '🌳', name: 'Ensemble' },
  v4_dl: { icon: '🔗', name: 'V4 Dedup DL' },
};

function renderModelSwitcher(comparisons, primaryRisk) {
  const switcherBtns = document.querySelectorAll('.model-switch-btn');

  // Mark unavailable buttons (model wasn't in comparisons)
  switcherBtns.forEach(btn => {
    const m = btn.dataset.model;
    if (!comparisons[m]) {
      btn.classList.add('unavailable');
      btn.title = 'Model unavailable (not trained or failed to load)';
    } else {
      btn.classList.remove('unavailable');
      btn.title = '';
    }
  });

  // Determine first available model to pre-select
  const available = Object.keys(comparisons);
  const firstModel = ['v3', 'dl', 'ensemble', 'v4_dl'].find(m => available.includes(m)) || available[0];

  // Pre-activate first button and display its panel
  switcherBtns.forEach(btn => {
    const m = btn.dataset.model;
    btn.classList.toggle('active', m === firstModel);
  });
  if (comparisons[firstModel]) renderMLPanel(comparisons[firstModel]);

  // Click handlers
  switcherBtns.forEach(btn => {
    // Clone to remove old listeners
    const fresh = btn.cloneNode(true);
    btn.parentNode.replaceChild(fresh, btn);

    fresh.addEventListener('click', () => {
      const m = fresh.dataset.model;
      if (fresh.classList.contains('unavailable')) return;

      document.querySelectorAll('.model-switch-btn').forEach(b => b.classList.remove('active'));
      fresh.classList.add('active');

      // Swap the ML panel to show this model's data
      renderMLPanel(comparisons[m]);
    });
  });
}


// ── Gauge (SVG arc) ────────────────────────────────────────────────────
function renderGauge(score, label) {
  const canvas = $('gauge-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cx = 90, cy = 88, r = 70;
  const startAngle = Math.PI;
  const endAngle = 2 * Math.PI;
  const totalAngle = endAngle - startAngle;
  const scoreAngle = startAngle + (score / 100) * totalAngle;

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Score arc
  const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
  gradient.addColorStop(0, '#22c55e');
  gradient.addColorStop(0.4, '#f59e0b');
  gradient.addColorStop(1, '#ef4444');

  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, scoreAngle);
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const angle = startAngle + (i / 10) * totalAngle;
    const ix = cx + (r - 18) * Math.cos(angle);
    const iy = cy + (r - 18) * Math.sin(angle);
    const ox = cx + (r - 8) * Math.cos(angle);
    const oy = cy + (r - 8) * Math.sin(angle);
    ctx.beginPath(); ctx.moveTo(ix, iy); ctx.lineTo(ox, oy);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1.5; ctx.stroke();
  }

  // Score text
  $('gauge-score').textContent = score;
  const riskColors = { 'Clean': '#22c55e', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444' };
  const rl = $('gauge-risk-label');
  rl.textContent = label;
  rl.style.color = riskColors[label] || '#94a3b8';
}

// ── Issues ─────────────────────────────────────────────────────────────
function renderIssues(issues) {
  const list = $('issues-list');

  // Severity icon map
  const sevIcon = { CRITICAL: '🔴', HIGH: '🟠', MEDIUM: '🟡', LOW: '🟢' };

  function renderFiltered(filter) {
    const filtered = filter === 'ALL' ? issues : issues.filter(i => i.severity === filter);
    list.innerHTML = '';
    if (filtered.length === 0) {
      list.innerHTML = `<div class="no-issues-msg"><span>✅</span> No ${filter === 'ALL' ? '' : filter.toLowerCase() + ' '}issues found.</div>`;
      return;
    }
    filtered.forEach(issue => {
      const div = document.createElement('div');
      div.className = 'issue-item';
      div.innerHTML = `
        <span class="issue-sev-badge sev-${issue.severity}">${sevIcon[issue.severity] || ''} ${issue.severity}</span>
        <div class="issue-body">
          <div class="issue-message">${escHtml(issue.message)}</div>
          <div class="issue-meta">
            ${issue.line ? `<span class="issue-line">Line ${issue.line}</span>` : ''}
            ${issue.type ? `<span style="margin-left:8px;color:var(--clr-text-3);font-size:0.72rem;">${issue.type}</span>` : ''}
          </div>
          ${issue.suggestion ? `<div class="issue-suggest">💡 ${escHtml(issue.suggestion)}</div>` : ''}
        </div>`;
      list.appendChild(div);
    });
  }

  renderFiltered(currentIssueFilter);

  // Filter buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentIssueFilter = btn.dataset.sev;
      renderFiltered(currentIssueFilter);
    });
  });
}

// ── Suggestions ────────────────────────────────────────────────────────
function renderSuggestions(sugs) {
  const list = $('suggestions-list');
  list.innerHTML = '';
  if (sugs.length === 0) {
    list.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--clr-clean);font-size:.88rem;">✅ No improvement suggestions — code looks clean!</div>';
    return;
  }
  const catIcon = { 'Memory Management': '💧', 'Security': '🛡', 'Complexity': '🔄', 'Code Structure': '🏗', 'Performance': '⚡', 'Maintainability': '📐', 'General': '💡' };
  sugs.forEach(sug => {
    const div = document.createElement('div');
    div.className = 'suggestion-item';
    div.innerHTML = `
      <div class="sug-header">
        <span class="issue-sev-badge sev-${sug.severity}">${sug.severity}</span>
        <span class="sug-cat">${catIcon[sug.category] || '💡'} ${sug.category}</span>
      </div>
      <div class="sug-message">${escHtml(sug.message)}</div>
      <div class="sug-detail">${escHtml(sug.suggestion)}</div>
      ${sug.example ? `<pre class="sug-example">${escHtml(sug.example)}</pre>` : ''}`;
    list.appendChild(div);
  });
}

// ── Complexity Bar Chart ───────────────────────────────────────────────
function renderComplexityChart(funcComplexity) {
  const ctx = $('complexity-chart').getContext('2d');

  if (complexityChart) { complexityChart.destroy(); complexityChart = null; }

  if (!funcComplexity || !Array.isArray(funcComplexity) || !funcComplexity.length) return;

  const labels = funcComplexity.map(f => f.function || f.name || '__global__').filter(Boolean);
  const values = funcComplexity.map(f => Math.max(1, f.cyclomatic_complexity || 1));
  const colors = values.map(v =>
    v <= 5 ? 'rgba(99,102,241,0.8)' :
      v <= 10 ? 'rgba(245,158,11,0.8)' :
        'rgba(239,68,68,0.8)'
  );

  if (!labels.length || !values.length) {
    ctx.canvas.parentElement.innerHTML = '<div style="color:var(--clr-text-3);text-align:center;padding:2rem;">No complexity data available.</div>';
    return;
  }

  complexityChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Cyclomatic Complexity',
        data: values,
        backgroundColor: colors,
        borderColor: colors.filter(c => c).map(c => c ? c.replace('0.8', '1') : 'rgba(255,255,255,1)'),
        borderWidth: 1.5,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0d1117',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            label: ctx => ` CC: ${ctx.raw}  (${ctx.raw <= 5 ? 'Simple' : ctx.raw <= 10 ? 'Moderate' : 'Complex'})`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 11 }, maxRotation: 30 },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          min: 0,
          ticks: { color: '#64748b', stepSize: 1 },
          grid: { color: 'rgba(255,255,255,0.06)' },
        },
      },
    },
  });
}

// ── Functions Panel ────────────────────────────────────────────────────
function renderFunctions(functions, funcComplexity, cfgs) {
  const list = $('functions-list');
  list.innerHTML = '';

  const ccMap = {};
  if (Array.isArray(funcComplexity)) {
    funcComplexity.forEach(f => {
      if (f && f.function) ccMap[f.function] = f.cyclomatic_complexity || 1;
      if (f && f.name) ccMap[f.name] = f.cyclomatic_complexity || 1;
    });
  }

  if (!functions || functions.length === 0) {
    list.innerHTML = '<div style="color:var(--clr-text-3);font-size:.85rem;">No functions detected.</div>';
  } else {
    functions.forEach(fn => {
      if (!fn || !fn.name) return; // Skip invalid entries
      const cc = ccMap[fn.name] || 1;
      const ccClass = cc <= 5 ? 'fn-cc-low' : cc <= 10 ? 'fn-cc-mod' : 'fn-cc-high';
      const chip = document.createElement('div');
      chip.className = 'fn-chip';
      chip.innerHTML = `
        <span class="fn-name">${escHtml(fn.name)}()</span>
        <span class="fn-line">L${fn.line || 'N/A'}</span>
        <span class="fn-cc-badge ${ccClass}">CC:${cc}</span>`;
      list.appendChild(chip);
    });
  }

  // Mermaid dependency graph
  renderMermaidGraph(functions, cfgs);
}

function renderMermaidGraph(functions, cfgs) {
  const container = $('mermaid-graph');

  if (!functions || functions.length === 0) {
    container.innerHTML = '<div style="color:var(--clr-text-3);font-size:.82rem;">No functions to graph.</div>';
    return;
  }

  const safeName = n => {
    if (!n || typeof n !== 'string') return '_unknown_';
    return n.replace(/[^a-zA-Z0-9_]/g, '_');
  };
  let graph = 'graph LR\n';

  // If we have actual CFG generation data from the V3 backend:
  if (cfgs && cfgs.length > 0) {
    // Collect all unique function blocks first
    cfgs.forEach(cfg => {
      const gName = safeName(cfg.func_name);
      graph += `  ${gName}("${cfg.func_name}()")\n`;
      graph += `  style ${gName} fill:#1e1b4b,stroke:#06b6d4,stroke-width:2px,color:#e2e8f0\n`;

      // Look explicitly for Call Graph relations mapped by the Recursion Analyzer
      if (cfg.nodes) {
        for (let n_id in cfg.nodes) {
          const node = cfg.nodes[n_id];

          // Draw dependency arrows for explicit function invocations
          if (node.type === "FuncCall" && node.func_name) {
            const targetName = safeName(node.func_name);
            graph += `  ${gName} -->|Calls| ${targetName}\n`;
          }
        }
      }
    });
  } else {
    // Fallback: Disconnected nodes for Java / Basic tracking
    functions.forEach((fn, i) => {
      graph += `  ${safeName(fn.name)}["${fn.name}()  L${fn.line}"]\n`;
      graph += `  style ${safeName(fn.name)} fill:#1e1b4b,stroke:#6366f1,color:#e2e8f0\n`;
    });
  }

  const mermaidDiv = document.createElement('div');
  mermaidDiv.className = 'mermaid';
  mermaidDiv.textContent = graph;
  container.innerHTML = '';
  container.appendChild(mermaidDiv);

  try {
    mermaid.init({ theme: 'dark', themeVariables: { primaryColor: '#6366f1', edgeLabelBackground: '#0d1117' } }, mermaidDiv);
  } catch (e) {
    container.innerHTML = '<div style="color:var(--clr-text-3);font-size:.78rem;">Graph rendering skipped.</div>';
  }
}

// ── Tab Navigation ─────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    btn.classList.add('active');
    const panel = $(`tab-${btn.dataset.tab}`);
    if (panel) panel.classList.remove('hidden');

    // Re-render chart on tab switch (needed for responsive resize)
    if (btn.dataset.tab === 'complexity' && currentResults) {
      renderComplexityChart(currentResults.function_complexity || []);
    }
  });
});

// ── Utilities ──────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Init ───────────────────────────────────────────────────────────────
(function init() {
  mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'loose' });
  // Load C sample by default
  codeEditor.value = SAMPLES.C;
  updateLineNumbers();

  // ── Warmup: ping backend so ML model is pre-loaded before first upload
  // This prevents the "first request slow" cold-start problem.
  fetch(`${API_BASE}/health`, { method: 'GET' })
    .then(r => r.json())
    .then(() => {
      const indicator = document.querySelector('.server-status') || null;
      if (indicator) indicator.textContent = '🟢 Connected';
      console.log('[IntelliReview] Backend warmed up ✅');
    })
    .catch(() => {
      console.warn('[IntelliReview] Backend not reachable — start server with: python main.py');
      // Show a subtle warning banner
      const banner = document.createElement('div');
      banner.style.cssText = 'position:fixed;bottom:1rem;right:1rem;background:#ef4444;color:#fff;' +
        'padding:.5rem 1rem;border-radius:.5rem;font-size:.8rem;z-index:9999;';
      banner.textContent = '⚠️ Backend offline — run: python main.py';
      document.body.appendChild(banner);
      setTimeout(() => banner.remove(), 8000);
    });
})();
