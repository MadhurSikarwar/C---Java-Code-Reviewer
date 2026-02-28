/**
 * IntelliReview — Frontend Application Logic
 * Handles: code editor, API calls, risk gauge, Chart.js complexity chart, Mermaid graph
 */

const API_BASE = 'http://localhost:8000';

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
analyzeBtn.addEventListener('click', runAnalysis);
$('retry-btn').addEventListener('click', runAnalysis);

async function runAnalysis() {
  const code = codeEditor.value.trim();
  if (!code) { showError('Please paste or upload some code first.'); return; }

  const modelType = $('model-select').value;

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
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, language: currentLanguage, model_type: modelType }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    currentResults = data;
    await new Promise(r => setTimeout(r, 400)); // brief pause for final step
    steps[steps.length - 1].classList.add('done');
    await new Promise(r => setTimeout(r, 300));
    renderResults(data);
  } catch (err) {
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

  let report = `================================================================
INTELLIREVIEW — AI CODE ANALYSIS REPORT
================================================================
Language: ${r.language || currentLanguage}
Model Used: ${$('model-select').options[$('model-select').selectedIndex].text}
Time: ${new Date().toLocaleString()}

-- RISK SCORE --
Label: ${r.risk.label}
Score: ${r.risk.score}/100
Probabilities: Clean (${r.risk.probabilities.Clean}%), Moderate (${r.risk.probabilities['Moderate Risk']}%), High (${r.risk.probabilities['High Risk']}%)
`;

  if (r.risk.comparisons && Object.keys(r.risk.comparisons).length > 1) {
    report += `\n-- MODEL COMPARISONS --\n`;
    const names = { ensemble: 'Ensemble', dl: 'Massive CNN', v3: 'V3 Path-Sensitive' };
    for (const [mType, comp] of Object.entries(r.risk.comparisons)) {
      report += `* ${names[mType] || mType.toUpperCase()}:\n`;
      report += `  Label: ${comp.risk_label} | Score: ${comp.risk_score}/100\n`;
      report += `  Probabilities: Clean (${comp.probabilities.Clean}%), Mod (${comp.probabilities['Moderate Risk']}%), High (${comp.probabilities['High Risk']}%)\n`;
    }
  }

  report += `
-- METRICS --
Lines of Code:          ${metrics.lines_of_code}
Number of Functions:    ${metrics.num_functions}
Cyclomatic Complexity:  ${metrics.cyclomatic_complexity}
Time Complexity:        ${metrics.time_complexity}
Memory Leaks Detected:  ${metrics.memory_leak_count}
Unsafe C Functions:     ${metrics.unsafe_function_count}
Recursion Count:        ${metrics.recursion_count}

-- VULNERABILITY CATEGORIES --
Memory & Pointers:      ${r.risk.categories?.memory || 0}
Data Flow (SSA):        ${r.risk.categories?.data_flow || 0}
Architecture/Smells:    ${r.risk.categories?.architecture || 0}
Recursion Hazards:      ${r.risk.categories?.recursion || 0}

-- ISSUES DETECTED (${issues.length}) --\n`;

  if (issues.length === 0) {
    report += "✅ No issues detected.\n";
  } else {
    issues.forEach((i, idx) => {
      report += `[${idx + 1}] SEVERITY: ${i.severity}\n`;
      if (i.line) report += `    Line: ${i.line}\n`;
      if (i.type) report += `    Type: ${i.type}\n`;
      report += `    Message: ${i.message}\n`;
      if (i.suggestion) report += `    Fix: ${i.suggestion}\n`;
      report += "\n";
    });
  }

  report += `-- SUGGESTIONS (${suggestions.length}) --\n`;
  if (suggestions.length === 0) {
    report += "✅ No suggestions.\n";
  } else {
    suggestions.forEach((s, idx) => {
      report += `[${idx + 1}] ${s.category} (${s.severity})\n`;
      report += `    Message: ${s.message}\n`;
      report += `    Suggestion: ${s.suggestion}\n`;
      if (s.example) report += `    Example:\n${s.example.split('\\n').map(l => '      ' + l).join('\\n')}\n`;
      report += "\n";
    });
  }

  // Trigger file download
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
  $('analyze-btn-text') && ($('analyze-btn-text').textContent = 'Analyzing…');
}

function showError(msg) {
  emptyState.classList.add('hidden');
  loadingState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  errorState.classList.remove('hidden');
  $('error-message').textContent = msg;
  analyzeBtn.classList.remove('loading');
}

// ── Render Results ─────────────────────────────────────────────────────
function renderResults(data) {
  loadingState.classList.add('hidden');
  analyzeBtn.classList.remove('loading');
  resultsContent.classList.remove('hidden');

  const { metrics, risk, issues, suggestions, function_complexity, functions, cfgs } = data;

  // Risk Gauge & Confidence
  renderGauge(risk.score, risk.label);
  $('proba-clean').textContent = risk.probabilities['Clean'] + '%';
  $('proba-moderate').textContent = risk.probabilities['Moderate Risk'] + '%';
  $('proba-high').textContent = risk.probabilities['High Risk'] + '%';

  // Inject Generation 4 ML Confidence Penalty Output
  if ($('confidence-score')) {
    const conf = risk.confidence || 0;
    $('confidence-score').textContent = conf + '%';

    // Pick the most severe/top explanation for the UI
    const exList = risk.explanations || [];
    $('confidence-explain').textContent = exList.length > 0
      ? exList[exList.length - 1]
      : "Prediction bounds aligned successfully.";

    // Style penalty text if hard override
    if (conf === 100 && risk.label === "High Risk") {
      $('confidence-explain').style.color = "var(--clr-high)";
    } else if (conf < 70) {
      $('confidence-explain').style.color = "var(--clr-mod)";
    } else {
      $('confidence-explain').style.color = "var(--clr-text-3)";
    }
  }

  // Handle Comparisons from "All" mode
  const compCard = $('comparison-card');
  const compList = $('comparison-list');
  if (compCard && compList) {
    if (risk.comparisons && Object.keys(risk.comparisons).length > 1) {
      compCard.style.display = 'block';
      compList.innerHTML = '';
      const icons = { ensemble: '🌳', dl: '🧠', v3: '🔬' };
      const names = { ensemble: 'Ensemble RF', dl: 'Massive CNN', v3: 'V3 Path Engine' };

      for (const [mType, r] of Object.entries(risk.comparisons)) {
        const cls = r.risk_label === 'High Risk' ? 'var(--clr-high)' :
          r.risk_label === 'Moderate Risk' ? 'var(--clr-mod)' : 'var(--clr-clean)';
        compList.innerHTML += `
          <div style="background: rgba(255,255,255,0.03); padding: 0.8rem; border-radius: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.8rem;">
              <span style="font-size: 1.5rem;">${icons[mType] || '⚙️'}</span>
              <div>
                <div style="font-size: 0.85rem; color: var(--clr-text-3); text-transform: uppercase; letter-spacing: 0.5px;">${names[mType] || mType}</div>
                <div style="font-weight: 600;">Score: ${r.risk_score}/100</div>
              </div>
            </div>
            <div style="font-weight: 700; color: ${cls}; background: ${cls}22; padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">
              ${r.risk_label}
            </div>
          </div>
        `;
      }
    } else {
      compCard.style.display = 'none';
    }
  }

  // Metrics
  $('m-loc').textContent = metrics.lines_of_code;
  $('m-cc').textContent = metrics.cyclomatic_complexity;
  $('m-time').textContent = metrics.time_complexity;
  $('m-fn').textContent = metrics.num_functions;
  $('m-leak').textContent = metrics.memory_leak_count;
  $('m-unsafe').textContent = metrics.unsafe_function_count;

  // Warn on bad metrics
  if (metrics.memory_leak_count > 0) $('metric-leak').classList.add('warn');
  if (metrics.unsafe_function_count > 0) $('metric-unsafe').classList.add('warn');

  // Issues
  renderIssues(issues || []);
  $('tab-count-issues').textContent = (issues || []).length;

  // Suggestions
  const sugs = (suggestions && suggestions.suggestions) || [];
  renderSuggestions(sugs);
  $('tab-count-suggestions').textContent = sugs.length;

  // Complexity Chart
  renderComplexityChart(function_complexity || []);

  // Functions & CFG Graph
  renderFunctions(functions || [], function_complexity || [], cfgs || []);
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

  if (!funcComplexity.length) return;

  const labels = funcComplexity.map(f => f.function || '__global__');
  const values = funcComplexity.map(f => f.cyclomatic_complexity || 1);
  const colors = values.map(v =>
    v <= 5 ? 'rgba(99,102,241,0.8)' :
      v <= 10 ? 'rgba(245,158,11,0.8)' :
        'rgba(239,68,68,0.8)'
  );

  complexityChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Cyclomatic Complexity',
        data: values,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.8', '1')),
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
  funcComplexity.forEach(f => ccMap[f.function] = f.cyclomatic_complexity);

  if (functions.length === 0) {
    list.innerHTML = '<div style="color:var(--clr-text-3);font-size:.85rem;">No functions detected.</div>';
  } else {
    functions.forEach(fn => {
      const cc = ccMap[fn.name] || 1;
      const ccClass = cc <= 5 ? 'fn-cc-low' : cc <= 10 ? 'fn-cc-mod' : 'fn-cc-high';
      const chip = document.createElement('div');
      chip.className = 'fn-chip';
      chip.innerHTML = `
        <span class="fn-name">${escHtml(fn.name)}()</span>
        <span class="fn-line">L${fn.line}</span>
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

  const safeName = n => n.replace(/[^a-zA-Z0-9_]/g, '_');
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
