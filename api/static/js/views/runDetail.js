import { getRun, clarifyRun, cancelRun, streamUrl, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { emptyState, loadingRow } from '../components/card.js';
import { fmtMs, fmtNumber, fmtRelativeTime, titleCase, escapeHtml, prettyJson, statusColor } from '../format.js';
import { toastError, toastSuccess } from '../toast.js';
import { setUltraAmbient } from '../ambient.js';

let root = null;
let es = null;
let runUuid = null;
let data = { status: null, stages: [], artifacts: {}, clarification: null, mode: null };
let clarifySubmitting = false;

const ARTIFACT_ORDER = [
  'classification', 'planspec', 'draft', 'appraisal_report', 'fixed',
  'critique', 'verdict', 'final', 'final_validation', 'run',
];

export function mount(el, { runUuid: uuid }) {
  root = el;
  runUuid = uuid;
  data = { status: null, stages: [], artifacts: {}, clarification: null, mode: null };
  renderSkeleton();
  loadInitial();
}

export function unmount() {
  closeStream();
  setUltraAmbient(false);
  root = null;
  runUuid = null;
}

function closeStream() {
  if (es) {
    es.close();
    es = null;
  }
}

async function loadInitial() {
  try {
    const r = await getRun(runUuid);
    applyRunDetail(r);
    renderAll();
    openStream();
  } catch (err) {
    if (!root) return;
    document.getElementById('run-body').innerHTML = emptyState({
      iconName: 'alertTriangle',
      title: "Couldn't load this run",
      sub: err instanceof ApiError ? err.message : 'It may not exist, or the API base URL is wrong.',
    });
  }
}

function applyRunDetail(r) {
  data.status = r.status;
  data.stages = r.stages || [];
  data.artifacts = r.artifacts || {};
  data.clarification = r.clarification || null;
  if (r.artifacts && r.artifacts.run) {
    data.mode = r.artifacts.run.mode || data.mode;
  }
  setUltraAmbient(data.mode === 'ultra' || isUltraRun());
}

function isUltraRun() {
  return data.stages.some((s) => s.stage === 'model_swap') || data.mode === 'ultra';
}

function openStream() {
  closeStream();
  const isTerminal = ['complete', 'unresolvable', 'error', 'cancelled', 'interrupted'].includes(data.status);
  if (isTerminal) return; // nothing left to stream

  // Per the contract, reconnecting replays stages.log from the start —
  // so we reset our local list on every (re)connection to avoid duplicates.
  data.stages = [];
  renderTimeline();

  es = new EventSource(streamUrl(runUuid));
  es.onmessage = (evt) => {
    let payload;
    try { payload = JSON.parse(evt.data); } catch { return; }

    if (payload.type === 'clarification') {
      data.status = 'waiting_for_clarification';
      data.clarification = { question: payload.question };
      renderStatusBadge();
      renderClarification();
      renderTimeline();
      closeStream();
      return;
    }
    if (payload.type === 'complete') {
      data.status = payload.status;
      closeStream();
      renderStatusBadge();
      renderTimeline();
      refreshArtifacts();
      return;
    }
    if (payload.type === 'error') {
      toastError(payload.message || 'Stream error.');
      return;
    }
    // Otherwise: a stage log entry
    data.stages.push(payload);
    renderTimeline();
  };
  es.onerror = () => {
    // The browser's native EventSource auto-retry reuses this same instance
    // and would replay the stage backlog straight into our onmessage handler
    // without ever resetting data.stages, producing duplicate entries. Take
    // reconnection over explicitly instead: close this instance and re-run
    // openStream(), which resets state cleanly, same as any other reconnect.
    if (!es) return; // already closed deliberately elsewhere
    es.close();
    es = null;
    setTimeout(() => {
      if (!root || !runUuid) return; // navigated away in the meantime
      openStream();
    }, 1500);
  };
}

async function refreshArtifacts() {
  try {
    const r = await getRun(runUuid);
    applyRunDetail(r);
    renderAll();
  } catch {
    // keep whatever we already rendered
  }
}

function renderSkeleton() {
  root.innerHTML = `
    <div class="view-inner">
      <button class="btn-ghost" id="back-btn" style="width:fit-content;">${icon('arrowLeft')} All runs</button>
      <div class="view-header">
        <div>
          <h1 class="mono" style="font-size:16px;">${escapeHtml(runUuid)}</h1>
          <div class="view-desc" id="run-subtitle">Loading…</div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;" id="run-header-actions"></div>
      </div>
      <div id="run-body">${loadingRow('Loading run…')}</div>
    </div>
  `;
  document.getElementById('back-btn').addEventListener('click', () => { location.hash = '#/runs'; });
}

function renderAll() {
  if (!root) return;
  document.getElementById('run-body').innerHTML = `
    <div id="clarify-mount"></div>
    <section class="card">
      <div class="card-header" style="padding-bottom:6px;">
        <div class="card-icon-badge blue">${icon('layers')}</div>
        <div class="card-header-text">
          <div class="card-title">Stage progress</div>
          <div class="card-subtitle">Live via SSE — updates as each pipeline stage completes</div>
        </div>
      </div>
      <div class="card-body no-header" id="timeline-mount"></div>
    </section>
    <section id="artifacts-mount"></section>
  `;
  renderStatusBadge();
  renderTimeline();
  renderClarification();
  renderArtifacts();
}

function renderStatusBadge() {
  const sub = document.getElementById('run-subtitle');
  const actions = document.getElementById('run-header-actions');
  if (!sub || !actions) return;
  const color = statusColor(data.status);
  sub.innerHTML = `<span class="badge ${color}">${escapeHtml(titleCase(data.status || 'unknown'))}</span>`;

  const cancellable = ['running', 'pending', 'waiting_for_clarification'].includes(data.status);
  actions.innerHTML = cancellable
    ? `<button class="btn-pill red" id="cancel-btn">${icon('x')} Cancel run</button>`
    : '';
  const cancelBtn = document.getElementById('cancel-btn');
  if (cancelBtn) {
    cancelBtn.addEventListener('click', async () => {
      cancelBtn.disabled = true;
      try {
        await cancelRun(runUuid);
        toastSuccess('Cancel requested — the current model call will finish first.');
      } catch (err) {
        toastError(err instanceof ApiError ? err.message : 'Could not cancel.');
        cancelBtn.disabled = false;
      }
    });
  }
}

function renderTimeline() {
  const mount_ = document.getElementById('timeline-mount');
  if (!mount_) return;

  if (!data.stages.length && ['running', 'pending'].includes(data.status)) {
    mount_.innerHTML = loadingRow('Waiting for the first stage to complete…');
    return;
  }
  if (!data.stages.length) {
    mount_.innerHTML = emptyState({ iconName: 'layers', title: 'No stage activity recorded for this run' });
    return;
  }

  const isRunning = data.status === 'running' || data.status === 'pending';
  const nodes = data.stages.map((s, i) => stageNode(s, data.stages[i - 1], i === data.stages.length - 1 && !isRunning));

  let trailing = '';
  if (isRunning) {
    trailing = `
      <div class="stage-node">
        <div class="stage-node-rail">
          <span class="stage-node-dot active"></span>
        </div>
        <div class="stage-node-body">
          <div class="stage-node-head"><span class="stage-name text-secondary">Running next stage…</span></div>
        </div>
      </div>
    `;
  }

  mount_.innerHTML = nodes.join('') + trailing;
}

function stageNode(s, prev, isLast) {
  const isSwapEntry = s.stage === 'model_swap';
  const dotClass = isSwapEntry ? 'active' : (s.status === 'ok' || !s.status) ? '' : 'error';
  const showSwap = !isSwapEntry && prev && prev.model && s.model && prev.model !== s.model;

  const thinkBar = s.think_ratio > 0 ? `
    <span class="think-bar" title="Fraction of the response spent thinking">
      <span class="think-bar-track"><span class="think-bar-fill" style="width:${Math.round(s.think_ratio * 100)}%"></span></span>
      <span>${Math.round(s.think_ratio * 100)}% think</span>
    </span>` : '';

  const retryBadge = s.retries > 0 ? `<span class="badge orange">${s.retries} retr${s.retries === 1 ? 'y' : 'ies'}</span>` : '';
  const statusBadge = (s.status && s.status !== 'ok') ? `<span class="badge red">${escapeHtml(s.status)}</span>` : '';

  return `
    <div class="stage-node">
      <div class="stage-node-rail">
        <span class="stage-node-dot ${dotClass}"></span>
        ${isLast ? '' : '<span class="stage-node-line"></span>'}
      </div>
      <div class="stage-node-body">
        ${showSwap ? `<div class="stage-swap">${icon('swap')} swapped from ${escapeHtml(prev.model)}</div>` : ''}
        <div class="stage-node-head">
          <span class="stage-name">${escapeHtml(titleCase(isSwapEntry ? 'model swap' : s.stage))}</span>
          <span class="stage-model-badge mono">${escapeHtml(s.model || '—')}</span>
          ${retryBadge}${statusBadge}
        </div>
        <div class="stage-node-meta">
          <span class="meta-item"><strong>${fmtNumber(s.tokens_in)}</strong> in / <strong>${fmtNumber(s.tokens_out)}</strong> out</span>
          <span class="meta-item">${fmtMs(s.latency_ms)}</span>
          ${s.load_ms ? `<span class="meta-item">load ${fmtMs(s.load_ms)}</span>` : ''}
          ${s.ttft_ms ? `<span class="meta-item">ttft ${fmtMs(s.ttft_ms)}</span>` : ''}
          ${thinkBar}
        </div>
      </div>
    </div>
  `;
}

function renderClarification() {
  const mount_ = document.getElementById('clarify-mount');
  if (!mount_) return;
  if (data.status !== 'waiting_for_clarification' || !data.clarification) {
    mount_.innerHTML = '';
    return;
  }
  mount_.innerHTML = `
    <div class="clarify-card">
      <div style="display:flex;align-items:center;gap:8px;">
        <span class="badge purple">Needs your input</span>
      </div>
      <div class="clarify-question">${escapeHtml(data.clarification.question)}</div>
      <form class="clarify-form" id="clarify-form">
        <textarea class="textarea-input" id="clarify-answer" placeholder="Your answer…" required></textarea>
        <button type="submit" class="btn btn-primary" id="clarify-submit">Answer</button>
      </form>
    </div>
  `;
  document.getElementById('clarify-form').addEventListener('submit', onClarifySubmit);
}

async function onClarifySubmit(e) {
  e.preventDefault();
  if (clarifySubmitting) return;
  const answer = document.getElementById('clarify-answer').value.trim();
  if (!answer) return;
  clarifySubmitting = true;
  const btn = document.getElementById('clarify-submit');
  btn.disabled = true;
  btn.textContent = 'Sending…';
  try {
    await clarifyRun(runUuid, answer);
    data.status = 'running';
    data.clarification = null;
    toastSuccess('Answer sent — run resumed.');
    renderStatusBadge();
    renderClarification();
    openStream();
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Could not send answer.');
    btn.disabled = false;
    btn.textContent = 'Answer';
  } finally {
    clarifySubmitting = false;
  }
}

function renderArtifacts() {
  const mount_ = document.getElementById('artifacts-mount');
  if (!mount_) return;
  const keys = ARTIFACT_ORDER.filter((k) => data.artifacts[k] !== undefined);
  const extra = Object.keys(data.artifacts).filter((k) => !ARTIFACT_ORDER.includes(k));
  const allKeys = [...keys, ...extra];

  if (!allKeys.length) {
    mount_.innerHTML = '';
    return;
  }

  mount_.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:10px;">
      <div class="view-desc" style="margin-top:4px;">Artifacts</div>
      ${allKeys.map((k, i) => artifactBlock(k, data.artifacts[k], i === 0)).join('')}
    </div>
  `;
  mount_.querySelectorAll('[data-artifact-toggle]').forEach((headBtn) => {
    headBtn.addEventListener('click', () => {
      const body = headBtn.nextElementSibling;
      const open = headBtn.getAttribute('aria-expanded') === 'true';
      headBtn.setAttribute('aria-expanded', String(!open));
      body.style.display = open ? 'none' : 'block';
    });
  });
}

function artifactBlock(key, value, openByDefault) {
  return `
    <div class="artifact-block">
      <button class="artifact-block-head" data-artifact-toggle aria-expanded="${openByDefault}">
        <span>${escapeHtml(titleCase(key))}</span>
        <span class="chev">${icon('chevronRight')}</span>
      </button>
      <div class="artifact-block-body" style="display:${openByDefault ? 'block' : 'none'};">${escapeHtml(prettyJson(value))}</div>
    </div>
  `;
}
