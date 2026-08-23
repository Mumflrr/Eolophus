import { startRun, listRuns, cancelRun, listPipelines, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { cardHeader, emptyState, loadingRow } from '../components/card.js';
import { fmtNumber, fmtDuration, fmtRelativeTime, runDurationSeconds, statusColor, titleCase, escapeHtml } from '../format.js';
import { toastError, toastSuccess } from '../toast.js';
import { onHealthChange, getLastHealth } from '../components/health.js';
import { setUltraAmbient } from '../ambient.js';
import { setLaunching } from '../components/quickRun.js';

let root = null;
let pollTimer = null;
let unsubHealth = null;
let fetching = false;
let submitting = false;
let availablePipelines = [];

const state = {
  pipeline: '',       // '' = built-in pipeline; else a custom pipeline name
  mode: '',          // '' = auto
  taskType: '',       // '' = auto
  noEnsemble: false,
  useSearch: false,
  ultraAck: false,
};

const MODES = [
  { id: '', label: 'Auto' },
  { id: 'short', label: 'Short' },
  { id: 'long', label: 'Long' },
  { id: 'ultra', label: 'Ultra' },
];
const TASK_TYPES = [
  { id: '', label: 'Auto' },
  { id: 'coding', label: 'Coding' },
  { id: 'ideation', label: 'Ideation' },
  { id: 'mixed', label: 'Mixed' },
  { id: 'describe', label: 'Describe' },
];

export function mount(el) {
  root = el;
  submitting = false;
  render();
  loadRuns();
  loadPipelines();
  pollTimer = setInterval(loadRuns, 4000);
  unsubHealth = onHealthChange(() => renderQueueBanner());
  setUltraAmbient(state.mode === 'ultra');
}

export function unmount() {
  clearInterval(pollTimer);
  pollTimer = null;
  if (unsubHealth) unsubHealth();
  setUltraAmbient(false);
  root = null;
}

function render() {
  root.innerHTML = `
    <div class="view-inner">
      <div class="view-header">
        <div>
          <h1>Submit &amp; monitor runs</h1>
          <div class="view-desc">One pipeline run executes at a time on this rig — new tasks queue behind whatever's active.</div>
        </div>
      </div>

      <div id="queue-banner"></div>

      <section class="card" id="submit-card">
        ${cardHeader({ icon: 'play', color: 'blue', title: 'New run', subtitle: 'Submitted tasks run in the background' })}
        <div class="card-body no-header">
          <form id="submit-form" novalidate>
            <label class="field-label" for="task-input">Task</label>
            <textarea id="task-input" class="textarea-input" placeholder="Describe what you want the pipeline to do…" required></textarea>
            <div class="text-tertiary" style="font-size:11px;margin-top:6px;">
              Testing against the mock server? Include “clarify”, “fail”, or “slow” in the task text to exercise those flows.
            </div>

            <div style="height:16px"></div>
            <label class="field-label">Pipeline</label>
            <div id="pipeline-picker"></div>

            <div id="built-in-only-fields">
              <div style="height:16px"></div>
              <label class="field-label">Mode</label>
              <div class="segmented" id="mode-segmented" role="group" aria-label="Mode">
                ${MODES.map((m) => `<button type="button" data-mode="${m.id}" class="${m.id === 'ultra' ? 'danger' : ''} ${state.mode === m.id ? 'active' : ''}">${m.label}</button>`).join('')}
              </div>
              <div id="ultra-warning"></div>

              <div style="height:16px"></div>
              <label class="field-label">Task type</label>
              <div class="segmented" id="type-segmented" role="group" aria-label="Task type">
                ${TASK_TYPES.map((t) => `<button type="button" data-type="${t.id}" class="${state.taskType === t.id ? 'active' : ''}">${t.label}</button>`).join('')}
              </div>

              <div style="height:18px"></div>
              <div class="toggle-row" style="padding:4px 0;">
                <div class="card-row-label">
                  <span class="row-title">Skip ensemble</span>
                  <span class="row-sub">Bypass critique/synthesis passes for a faster, single-pass run</span>
                </div>
                <button type="button" class="toggle ${state.noEnsemble ? 'on' : ''}" id="toggle-ensemble" aria-pressed="${state.noEnsemble}" aria-label="Skip ensemble"></button>
              </div>
            </div>
            <div class="divider"></div>
            <div class="toggle-row" style="padding:12px 0 4px;">
              <div class="card-row-label">
                <span class="row-title">Web search</span>
                <span class="row-sub">Let the pipeline query SearXNG while working</span>
              </div>
              <button type="button" class="toggle ${state.useSearch ? 'on' : ''}" id="toggle-search" aria-pressed="${state.useSearch}" aria-label="Enable web search"></button>
            </div>

            <div style="height:18px"></div>
            <div style="display:flex;justify-content:flex-end;">
              <button type="submit" class="btn-pill blue" id="submit-btn">
                ${icon('play')} Start run
              </button>
            </div>
          </form>
        </div>
      </section>

      <section class="card">
        ${cardHeader({
          icon: 'runs', color: 'purple', title: 'Recent runs',
          actionHtml: `<button class="btn-icon" id="refresh-runs" title="Refresh now">${icon('refresh')}</button>`,
        })}
        <div id="run-list-body"></div>
      </section>
    </div>
  `;

  wireForm();
  renderQueueBanner();
  document.getElementById('refresh-runs').addEventListener('click', () => loadRuns(true));
}

async function loadPipelines() {
  try {
    availablePipelines = await listPipelines();
  } catch {
    availablePipelines = []; // non-fatal — picker just falls back to "built-in only"
  }
  if (root) renderPipelinePicker();
}

function wireForm() {
  document.getElementById('mode-segmented').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-mode]');
    if (!btn) return;
    state.mode = btn.dataset.mode;
    state.ultraAck = false;
    document.querySelectorAll('#mode-segmented button').forEach((b) => b.classList.toggle('active', b === btn));
    renderUltraWarning();
    setUltraAmbient(state.mode === 'ultra');
  });

  document.getElementById('type-segmented').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-type]');
    if (!btn) return;
    state.taskType = btn.dataset.type;
    document.querySelectorAll('#type-segmented button').forEach((b) => b.classList.toggle('active', b === btn));
  });

  document.getElementById('toggle-ensemble').addEventListener('click', (e) => {
    state.noEnsemble = !state.noEnsemble;
    e.currentTarget.classList.toggle('on', state.noEnsemble);
    e.currentTarget.setAttribute('aria-pressed', String(state.noEnsemble));
  });
  document.getElementById('toggle-search').addEventListener('click', (e) => {
    state.useSearch = !state.useSearch;
    e.currentTarget.classList.toggle('on', state.useSearch);
    e.currentTarget.setAttribute('aria-pressed', String(state.useSearch));
  });

  document.getElementById('submit-form').addEventListener('submit', onSubmit);
  renderUltraWarning();
  renderPipelinePicker();
}

function renderPipelinePicker() {
  const mount_ = document.getElementById('pipeline-picker');
  if (!mount_) return;
  mount_.innerHTML = `
    <select class="select-input" id="pipeline-select">
      <option value="" ${state.pipeline === '' ? 'selected' : ''}>Built-in pipeline</option>
      ${availablePipelines.map((p) => `<option value="${escapeHtml(p.name)}" ${state.pipeline === p.name ? 'selected' : ''}>${escapeHtml(p.name)} (${p.step_count} steps)</option>`).join('')}
    </select>
    ${availablePipelines.length === 0 ? `<div class="text-tertiary" style="font-size:11px;margin-top:6px;">No custom pipelines saved yet — build one on the Pipelines screen.</div>` : ''}
  `;
  document.getElementById('pipeline-select').addEventListener('change', (e) => {
    state.pipeline = e.target.value;
    toggleBuiltInFields();
  });
  toggleBuiltInFields();
}

// mode/task_type/no_ensemble only apply to the built-in pipeline — the
// contract says they're ignored once `pipeline` is set, so hide them
// rather than let the user configure settings that silently do nothing.
function toggleBuiltInFields() {
  const section = document.getElementById('built-in-only-fields');
  if (!section) return;
  section.style.display = state.pipeline ? 'none' : '';
  if (state.pipeline) {
    setUltraAmbient(false);
  } else {
    setUltraAmbient(state.mode === 'ultra');
  }
}

function renderUltraWarning() {
  const mount_ = document.getElementById('ultra-warning');
  if (!mount_) return;
  if (state.mode !== 'ultra') {
    mount_.innerHTML = '';
    return;
  }
  mount_.innerHTML = `
    <div style="height:10px"></div>
    <div class="disclaimer orange">
      ${icon('alertTriangle')}
      <div>
        <strong>Ultra mode swaps in the overnight deep-thinking model.</strong>
        It blocks interactive use of the rig, and a full run can take until morning to finish.
        Only start this if you don't need the GPU for anything else in the meantime.
        <label style="display:flex;align-items:center;gap:8px;margin-top:10px;cursor:pointer;">
          <input type="checkbox" id="ultra-ack" ${state.ultraAck ? 'checked' : ''} style="width:15px;height:15px;">
          I understand and want to proceed
        </label>
      </div>
    </div>
  `;
  document.getElementById('ultra-ack').addEventListener('change', (e) => {
    state.ultraAck = e.target.checked;
  });
}

function renderQueueBanner() {
  const mount_ = document.getElementById('queue-banner');
  if (!mount_) return;
  const h = getLastHealth();
  if (h && h.ok && h.queue_depth > 0) {
    mount_.innerHTML = `
      <div class="disclaimer blue">
        ${icon('clock')}
        <div>Queued — ${h.queue_depth} run${h.queue_depth === 1 ? '' : 's'} ahead of a new submission. Single GPU, one run at a time.</div>
      </div>
    `;
  } else {
    mount_.innerHTML = '';
  }
}

async function onSubmit(e) {
  e.preventDefault();
  if (submitting) return;
  const task = document.getElementById('task-input').value.trim();
  if (!task) {
    toastError('Describe the task before starting a run.');
    return;
  }
  if (!state.pipeline && state.mode === 'ultra' && !state.ultraAck) {
    toastError('Confirm you understand the Ultra mode trade-off first.');
    return;
  }

  submitting = true;
  setLaunching(true);
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Starting…`;

  try {
    const res = await startRun({
      task,
      pipeline: state.pipeline || null,
      mode: state.pipeline ? null : (state.mode || null),
      task_type: state.pipeline ? null : (state.taskType || null),
      no_ensemble: state.pipeline ? false : state.noEnsemble,
      use_search: state.useSearch,
    });
    toastSuccess('Run started.');
    submitting = false;
    setLaunching(false);
    location.hash = `#/runs/${res.run_uuid}`;
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Failed to start run.');
    submitting = false;
    setLaunching(false);
    btn.disabled = false;
    btn.innerHTML = `${icon('play')} Start run`;
  }
}

async function loadRuns(manual = false) {
  if (fetching) return;
  fetching = true;
  const body = document.getElementById('run-list-body');
  if (manual && body) body.innerHTML = loadingRow('Refreshing…');
  try {
    const runs = await listRuns(50);
    if (root) renderList(runs);
  } catch (err) {
    if (root && body && !body.dataset.hasContent) {
      body.innerHTML = emptyState({
        iconName: 'alertTriangle',
        title: "Can't load runs",
        sub: err instanceof ApiError ? err.message : 'Check the API base URL in Settings.',
      });
    }
  } finally {
    fetching = false;
  }
}

function renderList(runs) {
  const body = document.getElementById('run-list-body');
  if (!body) return;
  if (!runs.length) {
    body.innerHTML = emptyState({
      iconName: 'runs',
      title: 'No runs yet',
      sub: 'Submit a task above to start the pipeline.',
    });
    return;
  }
  body.dataset.hasContent = '1';
  body.innerHTML = `<div class="card-body no-header" style="padding-top:6px;">${runs.map(runRow).join('')}</div>`;
  body.querySelectorAll('[data-run-row]').forEach((rowEl) => {
    rowEl.addEventListener('click', () => {
      location.hash = `#/runs/${rowEl.dataset.runRow}`;
    });
  });
  body.querySelectorAll('[data-cancel]').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const uuid = btn.dataset.cancel;
      const isDelete = btn.dataset.action === 'delete';
      btn.disabled = true;
      try {
        await cancelRun(uuid);
        toastSuccess(isDelete ? 'Run deleted.' : 'Run cancelled.');
        loadRuns();
      } catch (err) {
        toastError(err instanceof ApiError ? err.message : (isDelete ? 'Could not delete run.' : 'Could not cancel run.'));
        btn.disabled = false;
      }
    });
  });
}

function runRow(r) {
  const color = statusColor(r.status);
  const cancellable = ['running', 'pending', 'waiting_for_clarification'].includes(r.status);
  const duration = r.total_latency_ms != null
    ? fmtDuration(r.total_latency_ms / 1000)
    : fmtDuration(runDurationSeconds(r));
  const actionBtn = cancellable
    ? `<button class="btn-icon" data-cancel="${r.run_uuid}" data-action="cancel" title="Cancel run" style="width:24px;height:24px;">${icon('x')}</button>`
    : `<button class="btn-icon" data-cancel="${r.run_uuid}" data-action="delete" title="Delete run" style="width:24px;height:24px;">${icon('trash')}</button>`;
  return `
    <div class="run-list-row" data-run-row="${r.run_uuid}">
      <span class="badge ${color}">${escapeHtml(titleCase(r.status))}</span>
      <div class="run-task">
        <span class="mono" style="color:var(--text-secondary);font-size:11.5px;">${r.run_uuid.slice(0, 8)}</span>
        &nbsp;·&nbsp; ${escapeHtml(titleCase(r.mode || 'auto'))} mode
        &nbsp;·&nbsp; ${escapeHtml(titleCase(r.task_type || 'auto'))}
        ${r.stage_reached ? `&nbsp;·&nbsp; <span class="text-tertiary">reached ${escapeHtml(r.stage_reached)}</span>` : ''}
      </div>
      <div class="run-meta-col">${r.total_tokens ? fmtNumber(r.total_tokens) + ' tok' : '—'}</div>
      <div class="run-meta-col">${duration}</div>
      <div class="run-meta-col" style="display:flex;align-items:center;gap:8px;justify-content:flex-end;">
        ${fmtRelativeTime(r.started_at)}
        ${actionBtn}
      </div>
    </div>
  `;
}