import { getApiBase, setApiBase, getSearxng, setSearxng, reloadConfig, getHealth, getBudgets, patchBudgets, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { cardHeader } from '../components/card.js';
import { escapeHtml, titleCase } from '../format.js';
import { toastError, toastSuccess } from '../toast.js';

let root = null;
let budgetsBaseline = {};
let budgetsSaving = false;

export function mount(el) {
  root = el;
  budgetsSaving = false;
  render();
  loadSearxng();
  loadBudgets();
}

export function unmount() {
  root = null;
}

function render() {
  root.innerHTML = `
    <div class="view-inner">
      <div class="view-header">
        <div>
          <h1>Settings</h1>
          <div class="view-desc">Connection and search configuration. No auth here by design — this is a localhost-only tool.</div>
        </div>
      </div>

      <section class="card">
        ${cardHeader({ icon: 'wifi', color: 'blue', title: 'API server', subtitle: 'Point the GUI at mock_server.py or server.py' })}
        <div class="card-body no-header">
          <label class="field-label" for="api-base-input">Base URL</label>
          <div style="display:flex;gap:8px;">
            <input type="text" class="text-input" id="api-base-input" value="${getApiBase()}" placeholder="http://localhost:8000">
            <button class="btn-ghost" id="test-connection-btn" style="white-space:nowrap;">${icon('wifi')} Test</button>
          </div>
          <div id="connection-result" style="margin-top:10px;"></div>
          <div style="height:14px"></div>
          <button class="btn btn-primary" id="save-api-base">Save</button>
        </div>
      </section>

      <section class="card">
        ${cardHeader({ icon: 'search', color: 'purple', title: 'SearXNG', subtitle: 'Local search instance used by the web-search node' })}
        <div class="card-body no-header">
          <label class="field-label" for="searxng-input">Instance URL</label>
          <div style="display:flex;gap:8px;">
            <input type="text" class="text-input" id="searxng-input" placeholder="http://localhost:8888">
            <button class="btn btn-primary" id="save-searxng-btn" style="white-space:nowrap;">Save</button>
          </div>
          <div class="view-desc" style="margin-top:8px;">Session-only — not written to disk, resets when the server restarts.</div>
        </div>
      </section>

      <section class="card">
        ${cardHeader({ icon: 'reload', color: 'orange', title: 'Config cache', subtitle: 'Bust in-memory caches after editing YAML by hand' })}
        <div class="card-body no-header">
          <button class="btn-ghost" id="reload-config-btn">${icon('reload')} Reload configuration</button>
        </div>
      </section>

      <section class="card">
        ${cardHeader({
          icon: 'sliders', color: 'blue', title: 'Built-in pipeline budgets',
          subtitle: 'Token budget per stage of the built-in classify→plan→draft→…→validate flow',
          actionHtml: `<button class="btn-ghost btn-sm" id="save-budgets" disabled>${icon('check')} Save</button>`,
        })}
        <div class="card-body no-header" id="budgets-list"><div class="loading-row"><span class="spinner"></span>Loading budgets…</div></div>
      </section>
    </div>
  `;

  document.getElementById('save-api-base').addEventListener('click', () => {
    const val = document.getElementById('api-base-input').value.trim();
    if (!val) return;
    setApiBase(val);
    toastSuccess('API base URL saved.');
  });

  document.getElementById('test-connection-btn').addEventListener('click', testConnection);
  document.getElementById('save-searxng-btn').addEventListener('click', saveSearxng);
  document.getElementById('reload-config-btn').addEventListener('click', onReload);
  document.getElementById('save-budgets').addEventListener('click', onSaveBudgets);
}

async function testConnection() {
  const resultEl = document.getElementById('connection-result');
  const inputVal = document.getElementById('api-base-input').value.trim();
  const prevBase = getApiBase();
  if (inputVal) setApiBase(inputVal);
  resultEl.innerHTML = `<div class="loading-row" style="padding:0;"><span class="spinner"></span> Testing…</div>`;
  try {
    const h = await getHealth();
    resultEl.innerHTML = `<div class="disclaimer blue">${icon('checkCircle')}<div>Connected. Hot model: <span class="mono">${h.hot_model || 'none'}</span></div></div>`;
  } catch (err) {
    resultEl.innerHTML = `<div class="disclaimer red">${icon('alertTriangle')}<div>${err instanceof ApiError ? err.message : 'Could not connect.'}</div></div>`;
    if (!inputVal) setApiBase(prevBase);
  }
}

async function loadSearxng() {
  try {
    const res = await getSearxng();
    const input = document.getElementById('searxng-input');
    if (input) input.value = res.url;
  } catch {
    // leave placeholder; API base is probably wrong, Test connection will surface it
  }
}

async function saveSearxng() {
  const val = document.getElementById('searxng-input').value.trim();
  if (!val) return;
  const btn = document.getElementById('save-searxng-btn');
  btn.disabled = true;
  try {
    await setSearxng(val);
    toastSuccess('SearXNG URL updated.');
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Could not update SearXNG URL.');
  } finally {
    btn.disabled = false;
  }
}

async function onReload() {
  const btn = document.getElementById('reload-config-btn');
  btn.disabled = true;
  try {
    await reloadConfig();
    toastSuccess('Configuration reloaded from disk.');
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Reload failed.');
  } finally {
    btn.disabled = false;
  }
}

async function loadBudgets() {
  const listEl = document.getElementById('budgets-list');
  try {
    const budgets = await getBudgets();
    budgetsBaseline = { ...budgets };
    if (!root) return;
    renderBudgets(budgets);
  } catch (err) {
    if (!root || !listEl) return;
    listEl.innerHTML = `<div class="disclaimer red">${icon('alertTriangle')}<div>${err instanceof ApiError ? err.message : "Can't load budgets."}</div></div>`;
  }
}

function renderBudgets(budgets) {
  const listEl = document.getElementById('budgets-list');
  const stages = Object.keys(budgets).sort();
  if (!stages.length) {
    listEl.innerHTML = '<div class="text-tertiary" style="font-size:12.5px;">No budgets configured.</div>';
    return;
  }
  listEl.innerHTML = `
    <div class="view-desc" style="margin-bottom:8px;"><span class="mono">-1</span> = unlimited (ultra mode only)</div>
    ${stages.map((stage) => `
      <div class="card-row">
        <div class="card-row-label"><span class="row-title mono">${escapeHtml(titleCase(stage))}</span></div>
        <div class="card-row-control">
          <input type="number" class="number-input" data-stage="${escapeHtml(stage)}" value="${budgets[stage]}" min="-1" step="1">
        </div>
      </div>
    `).join('')}
  `;
  listEl.querySelectorAll('[data-stage]').forEach((input) => {
    input.addEventListener('input', () => {
      input.classList.toggle('dirty', Number(input.value) !== budgetsBaseline[input.dataset.stage]);
      updateBudgetsSaveState();
    });
  });
  updateBudgetsSaveState();
}

function collectDirtyBudgets() {
  const dirty = {};
  document.querySelectorAll('#budgets-list [data-stage]').forEach((input) => {
    const v = Number(input.value);
    if (!Number.isInteger(v) || v < -1) return;
    if (v !== budgetsBaseline[input.dataset.stage]) dirty[input.dataset.stage] = v;
  });
  return dirty;
}

function updateBudgetsSaveState() {
  const btn = document.getElementById('save-budgets');
  if (!btn) return;
  btn.disabled = budgetsSaving || Object.keys(collectDirtyBudgets()).length === 0;
}

async function onSaveBudgets() {
  const dirty = collectDirtyBudgets();
  if (!Object.keys(dirty).length) return;
  budgetsSaving = true;
  const btn = document.getElementById('save-budgets');
  btn.disabled = true;
  const label = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> Saving…`;
  try {
    const res = await patchBudgets(dirty);
    budgetsBaseline = { ...res.budgets };
    toastSuccess(`Saved ${Object.keys(dirty).length} budget${Object.keys(dirty).length === 1 ? '' : 's'}.`);
    document.querySelectorAll('#budgets-list [data-stage]').forEach((input) => input.classList.remove('dirty'));
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Could not save budgets.');
  } finally {
    budgetsSaving = false;
    btn.innerHTML = label;
    updateBudgetsSaveState();
  }
}
