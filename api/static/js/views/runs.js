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
  profile: '',        // '' = auto. Renamed from `mode` — see PROFILES's comment.
  taskType: '',       // '' = auto
  // noEnsemble REMOVED: ensemble is now fully determined by profile choice
  // (short/medium have no critic_a/critic_b nodes in their node_set at all
  // — see routing.yaml's pipeline_profiles — while long/ultra always want
  // the full ensemble per complexity/trigger_on_profile). A manual skip
  // toggle sitting alongside profile selection was a second way to reach
  // the same decision; pick short or medium instead of toggling this off.
  useSearch: false,
  // human_in_the_loop (design doc §2.6): true (default, matches
  // RunRequest.human_in_the_loop's server-side default) — low confidence
  // still eventually halts for clarification once a stage's escalation
  // ladder is exhausted. false ("set-and-forget") — proceed best-effort
  // instead of halting.
  humanInTheLoop: true,
  ultraAck: false,
  attachments: [],    // [{ filename, content }] — text/code files, read client-side
  image: null,        // { filename, dataUrl } — single image, routed through vision_decode
};

// Mirrors _ALLOWED_IMAGE_TYPES in server.py — keep in sync.
const ACCEPTED_IMAGE_TYPES = 'image/png,image/jpeg,image/webp,image/gif';
const MAX_IMAGE_BYTES = 8 * 1024 * 1024; // 8MB — generous for a screenshot/photo

// Soft cap so a giant paste-in doesn't silently blow the context budget
// with no feedback — mirrors MAX_ATTACHMENT_CHARS server-side.
const MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024; // 2MB per file
const ACCEPTED_EXTENSIONS = '.py,.md,.txt,.json,.js,.jsx,.ts,.tsx,.yaml,.yml,.sh,.csv,.html,.css,.toml,.ini,.log,.rs,.go,.java,.c,.cpp,.h,.rb,.sql';

// RENAMED from MODES under the pipeline-profile design (see
// docs/pipeline-profile-escalation-design.md) — server.py's RunRequest
// already sends/reads requested_profile, not mode; mode is now
// informational-only on TaskClassification and no longer selects
// pipeline shape. Added "medium" as a real tier (previously only two
// existed: short/long). Mirrored in runDetail.js as CHAT_PROFILES — keep
// both in sync.
const PROFILES = [
  { id: '', label: 'Auto' },
  { id: 'short', label: 'Short' },
  { id: 'medium', label: 'Medium' },
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
  setUltraAmbient(state.profile === 'ultra');
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
            <label class="field-label">Attachments</label>
            <div id="attachment-list"></div>
            <label class="btn-ghost" for="attachment-input" style="width:fit-content;cursor:pointer;">
              ${icon('paperclip')} Attach files
            </label>
            <input type="file" id="attachment-input" accept="${ACCEPTED_EXTENSIONS}" multiple style="display:none;">
            <div class="text-tertiary" style="font-size:11px;margin-top:6px;">
              Text and code files only — content is included alongside your task.
            </div>

            <div style="height:16px"></div>
            <label class="field-label">Image</label>
            <div id="image-preview"></div>
            <label class="btn-ghost" for="image-input" style="width:fit-content;cursor:pointer;">
              ${icon('image')} Attach image
            </label>
            <input type="file" id="image-input" accept="${ACCEPTED_IMAGE_TYPES}" style="display:none;">
            <div class="text-tertiary" style="font-size:11px;margin-top:6px;">
              One image per run — analysed by the vision model before the rest of the pipeline runs. PNG, JPEG, WebP, or GIF, up to ${Math.round(MAX_IMAGE_BYTES / 1024 / 1024)}MB.
            </div>

            <div style="height:16px"></div>
            <label class="field-label">Pipeline</label>
            <div id="pipeline-picker"></div>

            <div id="built-in-only-fields">
              <div style="height:16px"></div>
              <label class="field-label">Profile</label>
              <div class="segmented" id="profile-segmented" role="group" aria-label="Profile">
                ${PROFILES.map((m) => `<button type="button" data-profile="${m.id}" class="${m.id === 'ultra' ? 'danger' : ''} ${state.profile === m.id ? 'active' : ''}">${m.label}</button>`).join('')}
              </div>
              <div id="ultra-warning"></div>

              <div style="height:16px"></div>
              <label class="field-label">Task type</label>
              <div class="segmented" id="type-segmented" role="group" aria-label="Task type">
                ${TASK_TYPES.map((t) => `<button type="button" data-type="${t.id}" class="${state.taskType === t.id ? 'active' : ''}">${t.label}</button>`).join('')}
              </div>

              <div style="height:18px"></div>
              <div class="toggle-row" style="padding:4px 0;" title="On (default): pause and ask before classify/plan calls a bigger model to try to resolve low confidence or a truncated result. Off: escalate automatically, and if it's still unsure after that, proceed with the best available result instead of stopping to ask.">
                <div class="card-row-label">
                  <span class="row-title">Ask if unsure</span>
                  <span class="row-sub">Confirm before escalating to a bigger model on low confidence or truncation</span>
                </div>
                <button type="button" class="toggle ${state.humanInTheLoop ? 'on' : ''}" id="toggle-human-in-the-loop" aria-pressed="${state.humanInTheLoop}" aria-label="Confirm before escalating on low confidence or truncation"></button>
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
  document.getElementById('profile-segmented').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-profile]');
    if (!btn) return;
    state.profile = btn.dataset.profile;
    state.ultraAck = false;
    document.querySelectorAll('#profile-segmented button').forEach((b) => b.classList.toggle('active', b === btn));
    renderUltraWarning();
    setUltraAmbient(state.profile === 'ultra');
  });

  document.getElementById('type-segmented').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-type]');
    if (!btn) return;
    state.taskType = btn.dataset.type;
    document.querySelectorAll('#type-segmented button').forEach((b) => b.classList.toggle('active', b === btn));
  });

  document.getElementById('toggle-human-in-the-loop').addEventListener('click', (e) => {
    state.humanInTheLoop = !state.humanInTheLoop;
    e.currentTarget.classList.toggle('on', state.humanInTheLoop);
    e.currentTarget.setAttribute('aria-pressed', String(state.humanInTheLoop));
  });
  document.getElementById('toggle-search').addEventListener('click', (e) => {
    state.useSearch = !state.useSearch;
    e.currentTarget.classList.toggle('on', state.useSearch);
    e.currentTarget.setAttribute('aria-pressed', String(state.useSearch));
  });

  document.getElementById('submit-form').addEventListener('submit', onSubmit);
  document.getElementById('attachment-input').addEventListener('change', onAttachmentChange);
  document.getElementById('image-input').addEventListener('change', onImageChange);
  renderUltraWarning();
  renderPipelinePicker();
  renderAttachmentList();
  renderImagePreview();
}

async function onImageChange(e) {
  const file = e.target.files && e.target.files[0];
  e.target.value = ''; // allow re-selecting the same file later
  if (!file) return;
  if (file.size > MAX_IMAGE_BYTES) {
    toastError(`${file.name} is too large (max ${Math.round(MAX_IMAGE_BYTES / 1024 / 1024)}MB).`);
    return;
  }
  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('read failed'));
      reader.readAsDataURL(file);
    });
    state.image = { filename: file.name, dataUrl };
    renderImagePreview();
  } catch {
    toastError(`Couldn't read ${file.name}.`);
  }
}

function renderImagePreview() {
  const mount_ = document.getElementById('image-preview');
  if (!mount_) return;
  if (!state.image) {
    mount_.innerHTML = '';
    return;
  }
  mount_.innerHTML = `
    <div class="attachment-row" style="display:flex;align-items:center;gap:8px;padding:6px 0;">
      <img src="${state.image.dataUrl}" alt="" style="width:40px;height:40px;object-fit:cover;border-radius:4px;">
      <span class="mono" style="font-size:12px;flex:1;">${escapeHtml(state.image.filename)}</span>
      <button type="button" class="btn-icon" id="remove-image" title="Remove" style="width:20px;height:20px;">${icon('x')}</button>
    </div>
  `;
  document.getElementById('remove-image').addEventListener('click', () => {
    state.image = null;
    renderImagePreview();
  });
}

async function onAttachmentChange(e) {
  const files = Array.from(e.target.files || []);
  e.target.value = ''; // allow re-selecting the same file later
  for (const file of files) {
    if (file.size > MAX_ATTACHMENT_BYTES) {
      toastError(`${file.name} is too large (max ${Math.round(MAX_ATTACHMENT_BYTES / 1024 / 1024)}MB).`);
      continue;
    }
    try {
      const content = await file.text();
      state.attachments.push({ filename: file.name, content });
    } catch {
      toastError(`Couldn't read ${file.name}.`);
    }
  }
  renderAttachmentList();
}

function renderAttachmentList() {
  const mount_ = document.getElementById('attachment-list');
  if (!mount_) return;
  if (!state.attachments.length) {
    mount_.innerHTML = '';
    return;
  }
  mount_.innerHTML = state.attachments.map((a, i) => `
    <div class="attachment-row" style="display:flex;align-items:center;gap:8px;padding:6px 0;">
      ${icon('file')}
      <span class="mono" style="font-size:12px;flex:1;">${escapeHtml(a.filename)}</span>
      <span class="text-tertiary" style="font-size:11px;">${fmtNumber(new Blob([a.content]).size)} B</span>
      <button type="button" class="btn-icon" data-remove-attachment="${i}" title="Remove" style="width:20px;height:20px;">${icon('x')}</button>
    </div>
  `).join('');
  mount_.querySelectorAll('[data-remove-attachment]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.removeAttachment);
      state.attachments.splice(idx, 1);
      renderAttachmentList();
    });
  });
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

// profile/task_type only apply to the built-in pipeline — the
// contract says they're ignored once `pipeline` is set, so hide them
// rather than let the user configure settings that silently do nothing.
// (no_ensemble no longer exists as a separate field — see PROFILES's
// comment on state above; ensemble presence is now fully implied by
// profile choice.)
function toggleBuiltInFields() {
  const section = document.getElementById('built-in-only-fields');
  if (!section) return;
  section.style.display = state.pipeline ? 'none' : '';
  if (state.pipeline) {
    setUltraAmbient(false);
  } else {
    setUltraAmbient(state.profile === 'ultra');
  }
}

function renderUltraWarning() {
  const mount_ = document.getElementById('ultra-warning');
  if (!mount_) return;
  if (state.profile !== 'ultra') {
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
  if (!state.pipeline && state.profile === 'ultra' && !state.ultraAck) {
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
      requested_profile: state.pipeline ? null : (state.profile || null),
      task_type: state.pipeline ? null : (state.taskType || null),
      use_search: state.useSearch,
      human_in_the_loop: state.pipeline ? true : state.humanInTheLoop,
      attachments: state.attachments,
      image: state.image ? { filename: state.image.filename, data_url: state.image.dataUrl } : null,
    });
    toastSuccess('Run started.');
    submitting = false;
    setLaunching(false);
    state.attachments = [];
    state.image = null;
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
        &nbsp;·&nbsp; ${escapeHtml(titleCase(r.profile || r.mode || 'auto'))}
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