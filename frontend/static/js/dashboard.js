/* LLM Eval CI/CD Dashboard — JavaScript */
'use strict';

const API = '';  // Same-origin; adjust to http://localhost:8000 for dev

let currentPage = 1;
let allRuns = [];
let PAGE_SIZE = 20;

// ─── Navigation ───────────────────────────────────────────────────────────────

const PAGE_TITLES = {
  'overview':   'Overview',
  'runs':       'Evaluation Runs',
  'run-detail': 'Run Details',
  'model-comp': 'Model Comparison',
  'rag-comp':   'RAG Comparison',
  'regression': 'Regression Analysis',
  'datasets':   'Dataset Versions',
  'cost':       'Cost / Quality',
  'gates':      'Quality Gates',
};

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`page-${name}`).classList.add('active');
  document.getElementById(`nav-${name}`).classList.add('active');
  document.getElementById('page-title').textContent = PAGE_TITLES[name] || name;

  // Lazy load page data
  if (name === 'overview') loadOverview();
  if (name === 'runs')     loadRunsPage();
  if (name === 'cost')     loadCostChart();
  if (name === 'gates')    loadGateHistory();
}

// ─── API Helpers ───────────────────────────────────────────────────────────────

async function apiFetch(path, opts = {}) {
  try {
    const res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error('API error:', e);
    return null;
  }
}

// ─── Overview ─────────────────────────────────────────────────────────────────

async function loadOverview() {
  const data = await apiFetch('/api/v1/evaluations?page_size=50');
  if (!data) return;

  allRuns = data.runs || [];
  const runs = allRuns;
  const total = data.total || runs.length;

  // Stat tiles
  document.getElementById('stat-total-runs').textContent = total;

  const qualities = runs.map(r => {
    const m = r.aggregate_metrics || {};
    return (m.faithfulness || 0) * 0.5 + (m.answer_relevance || 0) * 0.5;
  }).filter(v => v > 0);
  const avgQuality = qualities.length ? (qualities.reduce((a,b)=>a+b,0)/qualities.length) : 0;
  document.getElementById('stat-avg-quality').textContent = avgQuality ? `${(avgQuality*100).toFixed(1)}%` : '—';

  const latencies = runs.map(r => (r.aggregate_metrics||{}).p95_latency_ms).filter(Boolean);
  const avgLat = latencies.length ? Math.round(latencies.reduce((a,b)=>a+b,0)/latencies.length) : 0;
  document.getElementById('stat-avg-latency').textContent = avgLat ? `${avgLat}ms` : '—';

  const costs = runs.map(r => r.total_cost_usd / Math.max(r.total_cases,1)).filter(Boolean);
  const avgCost = costs.length ? costs.reduce((a,b)=>a+b,0)/costs.length : 0;
  document.getElementById('stat-avg-cost').textContent = avgCost ? `$${avgCost.toFixed(4)}` : '—';

  const failedGates = runs.filter(r => r.quality_gate_result === 'FAILED').length;
  document.getElementById('stat-failed-gates').textContent = failedGates;
  document.getElementById('stat-regressions').textContent = failedGates;

  // Table
  const tbody = document.getElementById('overview-runs-body');
  tbody.innerHTML = '';
  if (!runs.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--text-muted)">No runs yet — trigger an evaluation to get started</td></tr>';
    return;
  }

  runs.slice(0, 10).forEach(run => {
    const m = run.aggregate_metrics || {};
    const quality = ((m.faithfulness||0)*0.5 + (m.answer_relevance||0)*0.5);
    const gate = run.quality_gate_result;
    const gateHtml = gate
      ? `<span class="badge badge-${gate==='PASSED'?'green':'red'}">${gate}</span>`
      : '<span class="badge badge-blue">—</span>';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td onclick="loadRunById('${run.run_id}')" style="cursor:pointer;color:var(--accent)">${run.run_id}</td>
      <td>${run.model || '—'}</td>
      <td>${run.rag_version || '—'}</td>
      <td>${run.dataset_version || '—'}</td>
      <td>${quality ? (quality*100).toFixed(1)+'%' : '—'}</td>
      <td>${m.p95_latency_ms ? Math.round(m.p95_latency_ms)+'ms' : '—'}</td>
      <td>${run.total_cost_usd ? '$'+run.total_cost_usd.toFixed(4) : '—'}</td>
      <td>${gateHtml}</td>
      <td style="color:var(--text-muted)">${formatDate(run.timestamp)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ─── Runs Page ────────────────────────────────────────────────────────────────

async function loadRunsPage() {
  const data = await apiFetch(`/api/v1/evaluations?page=${currentPage}&page_size=${PAGE_SIZE}`);
  if (!data) return;

  allRuns = data.runs || [];
  const tbody = document.getElementById('runs-table-body');
  tbody.innerHTML = '';

  if (!allRuns.length) {
    tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:var(--text-muted)">No runs found</td></tr>';
    return;
  }

  allRuns.forEach(run => {
    const m = run.aggregate_metrics || {};
    const passRate = run.pass_rate != null ? `${(run.pass_rate*100).toFixed(0)}%` : '—';
    const faith = m.faithfulness ? `${(m.faithfulness*100).toFixed(1)}%` : '—';
    const hall = m.hallucination_rate != null ? `${(m.hallucination_rate*100).toFixed(1)}%` : '—';
    const gate = run.quality_gate_result;
    const gateHtml = gate ? `<span class="badge badge-${gate==='PASSED'?'green':'red'}">${gate}</span>` : '—';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--accent);cursor:pointer" onclick="loadRunById('${run.run_id}')">${run.run_id}</td>
      <td>${run.experiment_id||'—'}</td>
      <td>${run.model||'—'}</td>
      <td>${run.rag_version||'—'}</td>
      <td>${run.dataset_version||'—'}</td>
      <td>${run.total_cases||0}</td>
      <td>${passRate}</td>
      <td>${faith}</td>
      <td style="color:${m.hallucination_rate>0.05?'var(--red)':'inherit'}">${hall}</td>
      <td>${gateHtml}</td>
      <td style="color:var(--text-muted)">${formatDate(run.timestamp)}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('page-indicator').textContent = `Page ${currentPage} of ${Math.ceil(data.total/PAGE_SIZE)}`;
}

function changePage(dir) {
  currentPage = Math.max(1, currentPage + dir);
  loadRunsPage();
}

// ─── Run Detail ───────────────────────────────────────────────────────────────

async function loadRunDetail() {
  const runId = document.getElementById('run-id-input').value.trim();
  if (!runId) return;
  await loadRunById(runId);
}

async function loadRunById(runId) {
  showPage('run-detail');
  document.getElementById('run-id-input').value = runId;

  const run = await apiFetch(`/api/v1/evaluations/${runId}`);
  if (!run) {
    document.getElementById('run-detail-content').style.display = 'none';
    return;
  }

  document.getElementById('run-detail-content').style.display = 'block';

  // Gate banner
  const gbc = document.getElementById('gate-banner-container');
  if (run.quality_gate_result) {
    const passed = run.quality_gate_result === 'PASSED';
    gbc.innerHTML = `<div class="gate-banner ${passed?'passed':'failed'}">
      ${passed ? '✓ QUALITY GATE: PASSED' : '✗ QUALITY GATE: FAILED — Deployment Blocked'}
    </div>`;
  } else {
    gbc.innerHTML = '';
  }

  // Metadata table
  const metaTable = document.getElementById('run-meta-table');
  const metaFields = ['run_id','experiment_id','model','prompt_version','rag_version',
                      'kb_version','dataset_version','git_sha','environment','total_cases',
                      'passed_cases','error_count','total_cost_usd'];
  metaTable.innerHTML = metaFields.map(k =>
    `<tr><td style="color:var(--text-muted)">${k}</td><td>${run[k] ?? '—'}</td></tr>`
  ).join('');

  // Metrics
  const mc = document.getElementById('run-metrics-container');
  const metrics = run.aggregate_metrics || {};
  if (!Object.keys(metrics).length) {
    mc.innerHTML = '<p style="color:var(--text-muted)">No metrics available</p>';
    return;
  }

  mc.innerHTML = Object.entries(metrics).map(([name, score]) => {
    const isLatency = name.includes('latency');
    const isCost = name.includes('cost');
    const isHall = name.includes('hallucination');
    const pct = isLatency || isCost ? null : Math.round(score * 100);
    const display = isLatency ? `${Math.round(score)}ms` : isCost ? `$${score.toFixed(4)}` : `${(score*100).toFixed(1)}%`;
    const good = isHall ? score <= 0.05 : isLatency ? score <= 1000 : score >= 0.85;
    const barWidth = isLatency ? Math.min(score/2000*100,100) : Math.min(score*100, 100);
    const barClass = good ? 'fill-green' : 'fill-red';

    return `<div class="metric-row">
      <div class="metric-name">${name}</div>
      <div class="metric-score" style="color:${good?'var(--green)':'var(--red)'}">${display}</div>
      <div class="metric-bar">
        <div class="progress-bar">
          <div class="progress-fill ${barClass}" style="width:${barWidth}%"></div>
        </div>
      </div>
    </div>`;
  }).join('');
}

// ─── Model Comparison ─────────────────────────────────────────────────────────

async function loadComparison() {
  const input = document.getElementById('compare-run-ids').value.trim();
  const runIds = input.split(',').map(s => s.trim()).filter(Boolean);
  if (runIds.length < 2) {
    alert('Enter at least 2 run IDs separated by commas');
    return;
  }

  const result = await apiFetch('/api/v1/compare', {
    method: 'POST',
    body: JSON.stringify({ run_ids: runIds }),
  });

  const container = document.getElementById('comparison-result');
  if (!result || !result.rows) {
    container.innerHTML = '<p style="color:var(--red)">Comparison failed — check run IDs</p>';
    return;
  }

  const rows = result.rows.sort((a,b) => b.quality_score - a.quality_score);
  container.innerHTML = `
    <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">
      <span class="badge badge-green">Best Quality: ${result.best_quality}</span>
      <span class="badge badge-blue">Best Latency: ${result.best_latency}</span>
      <span class="badge badge-yellow">Lowest Cost: ${result.lowest_cost}</span>
      <span class="badge badge-blue">Best Balanced: ${result.best_balanced}</span>
    </div>
    <table class="data-table">
      <thead><tr>
        <th>Model</th><th>Quality</th><th>Faithfulness</th>
        <th>Answer Rel.</th><th>Hallucination</th><th>P95 (ms)</th>
        <th>Cost/Q</th><th>Weighted ↑</th>
      </tr></thead>
      <tbody>
        ${rows.map(r => `<tr>
          <td style="font-weight:600">${r.model}</td>
          <td>${(r.quality_score*100).toFixed(1)}%</td>
          <td>${(r.faithfulness*100).toFixed(1)}%</td>
          <td>${(r.answer_relevance*100).toFixed(1)}%</td>
          <td style="color:${r.hallucination_rate>0.05?'var(--red)':'var(--green)'}">${(r.hallucination_rate*100).toFixed(1)}%</td>
          <td>${Math.round(r.p95_latency_ms)}ms</td>
          <td>$${r.cost_per_query_usd.toFixed(4)}</td>
          <td style="color:var(--accent);font-weight:600">${(r.weighted_score*100).toFixed(1)}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

// ─── Regression ───────────────────────────────────────────────────────────────

async function analyzeRegression() {
  const baselineId = document.getElementById('baseline-id-input').value.trim();
  const candidateId = document.getElementById('candidate-id-input').value.trim();
  if (!baselineId || !candidateId) return;

  const result = document.getElementById('regression-result');
  result.innerHTML = '<p style="color:var(--text-muted)">Analyzing...</p>';

  // Fetch both runs and do client-side delta
  const [b, c] = await Promise.all([
    apiFetch(`/api/v1/evaluations/${baselineId}`),
    apiFetch(`/api/v1/evaluations/${candidateId}`),
  ]);

  if (!b || !c) {
    result.innerHTML = '<p style="color:var(--red)">Could not load one or both runs</p>';
    return;
  }

  const bm = b.aggregate_metrics || {};
  const cm = c.aggregate_metrics || {};
  const lowerBetter = new Set(['hallucination_rate','p50_latency_ms','p95_latency_ms','cost_per_query']);
  const allMetrics = new Set([...Object.keys(bm), ...Object.keys(cm)]);
  const rows = [];

  for (const metric of allMetrics) {
    const bs = bm[metric], cs = cm[metric];
    if (bs == null || cs == null) continue;
    const delta = cs - bs;
    const deltaPct = bs !== 0 ? (delta / Math.abs(bs)) * 100 : 0;
    const isWorse = lowerBetter.has(metric) ? deltaPct > 5 : deltaPct < -5;
    rows.push({ metric, bs, cs, deltaPct, isWorse });
  }

  const regressions = rows.filter(r => r.isWorse);
  const overallDelta = rows.length ? rows.reduce((s,r)=>s+r.deltaPct,0)/rows.length : 0;

  result.innerHTML = `
    <div class="gate-banner ${regressions.length?'failed':'passed'}" style="margin-bottom:16px">
      ${regressions.length ? `⚠ ${regressions.length} Potential Regression Area(s) Detected` : '✓ No Significant Regressions'}
    </div>
    <p style="color:var(--text-muted);font-size:12px;margin-bottom:16px">
      Note: These are potential regression areas based on metric deltas — not definitive causal attribution.
    </p>
    <table class="data-table">
      <thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta %</th><th>Status</th></tr></thead>
      <tbody>
        ${rows.map(r => `<tr>
          <td>${r.metric}</td>
          <td style="font-family:var(--mono)">${r.bs.toFixed(4)}</td>
          <td style="font-family:var(--mono)">${r.cs.toFixed(4)}</td>
          <td style="color:${r.deltaPct>=0?'var(--green)':'var(--red)'};font-family:var(--mono)">${r.deltaPct>=0?'+':''}${r.deltaPct.toFixed(1)}%</td>
          <td>${r.isWorse ? '<span class="badge badge-red">Regression</span>' : '<span class="badge badge-green">OK</span>'}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

// ─── Quality Gate ─────────────────────────────────────────────────────────────

async function evaluateGate() {
  const runId = document.getElementById('gate-run-id').value.trim();
  if (!runId) return;

  const result = await apiFetch('/api/v1/quality-gate', {
    method: 'POST',
    body: JSON.stringify({ run_id: runId }),
  });

  const container = document.getElementById('gate-result');
  if (!result) {
    container.innerHTML = '<p style="color:var(--red)">Gate evaluation failed</p>';
    return;
  }

  const passed = result.passed;
  container.innerHTML = `
    <div class="gate-banner ${passed?'passed':'failed'}">
      ${passed ? '✓ QUALITY GATE: PASSED' : '✗ QUALITY GATE: FAILED'}
    </div>
    ${result.failing_rules && result.failing_rules.length ? `
    <p style="font-weight:600;margin-bottom:8px;color:var(--red)">Failing Rules:</p>
    ${result.failing_rules.map(r => `
      <div style="padding:8px 12px;background:rgba(239,68,68,0.08);border-left:3px solid var(--red);
                  border-radius:4px;margin-bottom:6px;font-size:12px;font-family:var(--mono)">
        ✗ ${r.metric}: ${r.actual_score?.toFixed(4)} (${r.rule_type} ${r.threshold})
      </div>`).join('')}` : ''}
    <p style="color:var(--text-muted);font-size:11px;margin-top:12px">
      ${result.failing_count}/${result.total_rules} rules failed
    </p>`;
}

async function loadGateHistory() {
  const data = await apiFetch('/api/v1/evaluations?page_size=20');
  const tbody = document.getElementById('gate-history-body');
  tbody.innerHTML = '';
  if (!data?.runs?.length) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">No gate decisions recorded yet</td></tr>';
    return;
  }
  const gated = data.runs.filter(r => r.quality_gate_result);
  if (!gated.length) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">No gate decisions yet</td></tr>';
    return;
  }
  gated.forEach(run => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--accent)">${run.run_id}</td>
      <td><span class="badge badge-${run.quality_gate_result==='PASSED'?'green':'red'}">${run.quality_gate_result}</span></td>
      <td>—</td>
      <td style="color:var(--text-muted)">${formatDate(run.timestamp)}</td>`;
    tbody.appendChild(tr);
  });
}

// ─── Cost Chart ───────────────────────────────────────────────────────────────

async function loadCostChart() {
  const data = await apiFetch('/api/v1/evaluations?page_size=50');
  const canvas = document.getElementById('cost-quality-chart');
  const ctx = canvas.getContext('2d');
  const runs = data?.runs || [];

  const W = canvas.width, H = canvas.height;
  const pad = { top:20, right:20, bottom:40, left:60 };

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0f1117';
  ctx.fillRect(0, 0, W, H);

  // Draw axes
  ctx.strokeStyle = '#2d3148';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, H - pad.bottom);
  ctx.lineTo(W - pad.right, H - pad.bottom);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#8892a4';
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Cost per Query (USD)', W/2, H - 4);
  ctx.save();
  ctx.translate(14, H/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('Quality Score', 0, 0);
  ctx.restore();

  if (!runs.length) {
    ctx.fillStyle = '#4a5568';
    ctx.textAlign = 'center';
    ctx.font = '13px Inter';
    ctx.fillText('No run data — trigger an evaluation first', W/2, H/2);
    return;
  }

  // Plot points
  const points = runs.map(run => {
    const m = run.aggregate_metrics || {};
    const q = (m.faithfulness||0)*0.5 + (m.answer_relevance||0)*0.5;
    const cost = run.total_cost_usd / Math.max(run.total_cases, 1);
    return { run_id: run.run_id, quality: q, cost, model: run.model };
  }).filter(p => p.quality > 0);

  if (!points.length) return;

  const maxCost = Math.max(...points.map(p => p.cost)) * 1.1 || 0.01;
  const minQ = Math.min(...points.map(p => p.quality)) * 0.95;
  const maxQ = Math.max(...points.map(p => p.quality)) * 1.02 || 1;

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const colors = ['#4f6ef7','#22c55e','#f59e0b','#ef4444','#a78bfa'];
  points.forEach((p, i) => {
    const x = pad.left + (p.cost / maxCost) * plotW;
    const y = pad.top + (1 - (p.quality - minQ) / (maxQ - minQ)) * plotH;

    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = colors[i % colors.length];
    ctx.fill();

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(p.model || p.run_id.slice(0,8), x, y - 12);
  });

  // Cost/quality table
  const tableContainer = document.getElementById('cost-quality-table');
  tableContainer.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Model/Run</th><th>Quality</th><th>Cost/Query</th><th>Quality/$</th></tr></thead>
      <tbody>${points.map((p,i) => `<tr>
        <td style="color:${colors[i%colors.length]}">${p.model||p.run_id}</td>
        <td>${(p.quality*100).toFixed(1)}%</td>
        <td>$${p.cost.toFixed(4)}</td>
        <td>${p.cost > 0 ? (p.quality/p.cost).toFixed(0) : '∞'}</td>
      </tr>`).join('')}</tbody>
    </table>`;
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function formatDate(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleString('en-US', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' });
  } catch { return ts; }
}

async function refreshData() {
  const activePage = document.querySelector('.page.active')?.id?.replace('page-', '');
  if (activePage) showPage(activePage);
}

async function triggerRun() {
  const res = await apiFetch('/api/v1/evaluations/run', {
    method: 'POST',
    body: JSON.stringify({ config_path: 'configs/evaluation.yaml', environment: 'local' }),
  });
  if (res) {
    alert(`Evaluation started: ${res.run_id}`);
    loadOverview();
  } else {
    alert('Failed to start evaluation — check API connection');
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  loadOverview();
});
