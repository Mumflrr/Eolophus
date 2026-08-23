import { listModels, loadModel, unloadModel, reassignRole, ALL_ROLES, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { cardHeader, emptyState, loadingRow } from '../components/card.js';
import { escapeHtml, titleCase, fmtNumber } from '../format.js';
import { toast, toastError, toastSuccess } from '../toast.js';

let root = null;
let pollTimer = null;
let models = [];
let busy = false; // true while a load/unload is in flight — locks all controls
const roleOverrides = {}; // role -> modelId, optimistic client-side layer (see note below)

export function mount(el) {
  root = el;
  render();
  load();
  pollTimer = setInterval(load, 15000);
}

export function unmount() {
  clearInterval(pollTimer);
  pollTimer = null;
  root = null;
}

// Consistent color per pipeline role so tags are scannable at a glance.
// Uses the same semantic palette as .badge / .btn-pill (blue/purple/green/red/orange/grey).
// Grouped by what the stage does in the pipeline (api_contract.md ALL_ROLES).
const ROLE_COLORS = {
  // input understanding
  vision_decode: 'orange',
  classify: 'orange',
  describe: 'orange',

  // generation / drafting
  ideation: 'purple',
  plan: 'purple',
  draft: 'blue',
  draft_short: 'blue',
  bugfix: 'blue',

  // review / verification
  appraise: 'green',
  critic_a: 'green',
  critic_b: 'green',
  validate: 'green',
  final_validate: 'green',
  gatekeeper: 'green',

  // synthesis / distillation
  synthesis_complex: 'red',
  synthesis_simple: 'red',
  distill: 'red',

  // chess-specific
  chess_fast: 'grey',
  chess_slow: 'grey',

  // "ultra" tier — mirrors its base role's color
  ultra_plan: 'purple',
  ultra_draft: 'blue',
  ultra_appraise: 'green',
  ultra_critic: 'green',
};
const ROLE_COLOR_FALLBACK = 'grey';

function roleColor(role) {
  return ROLE_COLORS[role] || ROLE_COLOR_FALLBACK;
}


function render() {
  root.innerHTML = `
    <div class="view-inner">
      <div class="view-header">
        <div>
          <h1>Model management</h1>
          <div class="view-desc">Only one model is ever loaded at a time on this single-GPU rig.</div>
        </div>
        <button class="btn-ghost" id="refresh-models">${icon('refresh')} Refresh</button>
      </div>
      <div id="models-list">${loadingRow('Loading models…')}</div>

      <section class="card">
        ${cardHeader({ icon: 'sliders', color: 'purple', title: 'Role assignment', subtitle: 'Which model handles each pipeline stage' })}
        <div class="card-body no-header" id="role-list"></div>
      </section>
    </div>
  `;
  document.getElementById('refresh-models').addEventListener('click', () => load(true));
}

async function load(manual = false) {
  const listEl = document.getElementById('models-list');
  if (manual && listEl) listEl.innerHTML = loadingRow('Refreshing…');
  try {
    models = await listModels();
    if (!root) return;
    renderModels();
    renderRoles();
  } catch (err) {
    if (!root || !listEl) return;
    listEl.innerHTML = emptyState({
      iconName: 'alertTriangle',
      title: "Can't load models",
      sub: err instanceof ApiError ? err.message : 'Check the API base URL in Settings.',
    });
  }
}

function effectiveRoleMap() {
  const map = {};
  for (const role of ALL_ROLES) {
    const owner = models.find((m) => (m.roles || []).includes(role));
    if (owner) map[role] = owner.id;
  }
  Object.assign(map, roleOverrides);
  return map;
}

function renderModels() {
  const listEl = document.getElementById('models-list');
  if (!models.length) {
    listEl.innerHTML = emptyState({ iconName: 'cpu', title: 'No models configured' });
    return;
  }
  const roleMap = effectiveRoleMap();
  listEl.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:14px;">
      ${models.map((m) => modelCard(m, roleMap)).join('')}
    </div>
  `;
  listEl.querySelectorAll('[data-load]').forEach((btn) => btn.addEventListener('click', () => doLoad(btn.dataset.load)));
  listEl.querySelectorAll('[data-unload]').forEach((btn) => btn.addEventListener('click', () => doUnload(btn.dataset.unload)));
}

function modelCard(m, roleMap) {
  const effectiveRoles = ALL_ROLES.filter((r) => roleMap[r] === m.id);
  const rolesToShow = effectiveRoles.length ? effectiveRoles : (m.roles || []);
  const statusColor = m.is_hot ? 'green' : 'grey';
  const statusLabel = m.is_hot ? 'Loaded · hot' : (m.status === 'loaded' ? 'Loaded' : 'Unloaded');

  return `
    <div class="card">
      <div class="card-header" style="padding-bottom:10px;">
        <div class="card-icon-badge ${statusColor}">${icon('cpu')}</div>
        <div class="card-header-text">
          <div class="card-title">${escapeHtml(m.name)}</div>
          <div class="card-subtitle mono">${escapeHtml(m.id)} · ${escapeHtml(m.quant || '')} · ${escapeHtml(m.source || '')}</div>
        </div>
        <span class="badge ${statusColor}">
          <span class="status-dot ${statusColor}${m.is_hot ? ' pulsing' : ''}" style="width:6px;height:6px;"></span>
          ${statusLabel}
        </span>
      </div>
      <div class="card-body no-header">
        <div class="tag-row">
          ${rolesToShow.map((r) => `<span class="tag ${roleColor(r)}">${escapeHtml(r)}</span>`).join('') || '<span class="text-tertiary" style="font-size:11.5px;">no roles assigned</span>'}
        </div>
        <div style="display:flex;gap:18px;margin-top:12px;font-size:11px;" class="mono text-tertiary">
          <span>ctx ${fmtNumber(m.context_len)}</span>
          <span>${escapeHtml(m.base_url || '')}</span>
          ${m.thinking_default ? '<span>thinking: on</span>' : ''}
          ${m.mtp ? '<span>MTP</span>' : ''}
        </div>
        <div style="display:flex;gap:8px;margin-top:14px;">
          <button class="btn-ghost model-action-btn" data-load="${m.id}" ${(busy || m.is_hot) ? 'disabled' : ''}>${icon('play', 'action-icon-blue')} Load</button>
          <button class="btn-ghost model-action-btn" data-unload="${m.id}" ${(busy || !m.is_hot) ? 'disabled' : ''}>${icon('power', 'action-icon-grey')} Unload</button>
        </div>
      </div>
    </div>
  `;
}

async function doLoad(modelId) {
  if (busy) return;
  busy = true;
  renderModels();
  toast(`Loading ${modelId}… this can take up to a few minutes on a cold start.`, 'info');
  try {
    await loadModel(modelId);
    toastSuccess(`${modelId} is now hot.`);
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : `Failed to load ${modelId}.`);
  } finally {
    busy = false;
    load();
  }
}

async function doUnload(modelId) {
  if (busy) return;
  busy = true;
  renderModels();
  try {
    await unloadModel(modelId);
    toastSuccess(`${modelId} unloaded.`);
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : `Failed to unload ${modelId}.`);
  } finally {
    busy = false;
    load();
  }
}

function renderRoles() {
  const roleEl = document.getElementById('role-list');
  if (!roleEl) return;
  const roleMap = effectiveRoleMap();
  roleEl.innerHTML = ALL_ROLES.map((role) => {
    const current = roleMap[role] || '';
    return `
      <div class="card-row role-picker-row">
        <div class="card-row-label">
          <span class="row-title mono">${escapeHtml(role)}</span>
        </div>
        <div class="card-row-control">
          <select class="select-input" data-role="${role}">
            <option value="" ${current === '' ? 'selected' : ''}>Unassigned</option>
            ${models.map((m) => `<option value="${m.id}" ${current === m.id ? 'selected' : ''}>${escapeHtml(m.name)}</option>`).join('')}
          </select>
        </div>
      </div>
    `;
  }).join('');
  roleEl.querySelectorAll('[data-role]').forEach((sel) => {
    sel.addEventListener('change', async (e) => {
      const role = sel.dataset.role;
      const modelId = e.target.value;
      if (!modelId) return;
      sel.disabled = true;
      try {
        await reassignRole(modelId, role);
        roleOverrides[role] = modelId;
        toastSuccess(`${titleCase(role)} → ${modelId}`);
        renderModels();
      } catch (err) {
        toastError(err instanceof ApiError ? err.message : 'Could not reassign role.');
      } finally {
        sel.disabled = false;
      }
    });
  });
}
