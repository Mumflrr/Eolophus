import { listPipelines, deletePipeline, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { cardHeader, emptyState, loadingRow } from '../components/card.js';
import { escapeHtml } from '../format.js';
import { toastError, toastSuccess } from '../toast.js';

let root = null;
let pipelines = [];
let confirmingDelete = null;

export function mount(el) {
  root = el;
  confirmingDelete = null;
  render();
  load();
}

export function unmount() {
  root = null;
}

function render() {
  root.innerHTML = `
    <div class="view-inner">
      <div class="view-header">
        <div>
          <h1>Pipelines</h1>
          <div class="view-desc">Custom step sequences beyond the built-in flow — existing nodes, freeform prompts, and decision branches.</div>
        </div>
        <button class="btn btn-primary" id="new-pipeline-btn">${icon('plus')} New pipeline</button>
      </div>
      <section class="card">
        ${cardHeader({ icon: 'git', color: 'purple', title: 'Saved pipelines' })}
        <div class="card-body no-header" id="pipeline-list">${loadingRow('Loading pipelines…')}</div>
      </section>
    </div>
  `;
  document.getElementById('new-pipeline-btn').addEventListener('click', () => {
    location.hash = '#/pipelines/new';
  });
}

async function load() {
  const listEl = document.getElementById('pipeline-list');
  try {
    pipelines = await listPipelines();
    if (!root) return;
    renderList();
  } catch (err) {
    if (!root || !listEl) return;
    listEl.innerHTML = emptyState({
      iconName: 'alertTriangle',
      title: "Can't load pipelines",
      sub: err instanceof ApiError ? err.message : 'Check the API base URL in Settings.',
    });
  }
}

function renderList() {
  const listEl = document.getElementById('pipeline-list');
  if (!pipelines.length) {
    listEl.innerHTML = emptyState({
      iconName: 'git',
      title: 'No custom pipelines yet',
      sub: 'Create one to define a step sequence beyond the built-in flow.',
    });
    return;
  }
  listEl.innerHTML = pipelines.map(pipelineRow).join('');
  listEl.querySelectorAll('[data-open]').forEach((el) => {
    el.addEventListener('click', () => { location.hash = `#/pipelines/${encodeURIComponent(el.dataset.open)}`; });
  });
  listEl.querySelectorAll('[data-ask-delete]').forEach((btn) => {
    btn.addEventListener('click', (e) => { e.stopPropagation(); confirmingDelete = btn.dataset.askDelete; renderList(); });
  });
  listEl.querySelectorAll('[data-cancel-delete]').forEach((btn) => {
    btn.addEventListener('click', (e) => { e.stopPropagation(); confirmingDelete = null; renderList(); });
  });
  listEl.querySelectorAll('[data-confirm-delete]').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const name = btn.dataset.confirmDelete;
      btn.disabled = true;
      try {
        await deletePipeline(name);
        pipelines = pipelines.filter((p) => p.name !== name);
        confirmingDelete = null;
        toastSuccess(`Deleted '${name}'.`);
        renderList();
      } catch (err) {
        toastError(err instanceof ApiError ? err.message : 'Could not delete pipeline.');
        btn.disabled = false;
      }
    });
  });
}

function pipelineRow(p) {
  const invalid = (p.description || '').startsWith('[INVALID:');
  const isConfirming = confirmingDelete === p.name;
  return `
    <div class="card-row pipeline-row" data-open="${escapeHtml(p.name)}">
      <div class="card-row-label">
        <span class="row-title">
          ${escapeHtml(p.name)}
          ${invalid ? '<span class="badge red" style="margin-left:8px;">invalid</span>' : ''}
        </span>
        <span class="row-sub">${escapeHtml(p.description || 'No description')}</span>
      </div>
      <div class="card-row-control">
        <span class="tag mono">${p.step_count} step${p.step_count === 1 ? '' : 's'}</span>
        <span class="tag mono">entry: ${escapeHtml(p.entry_step || '—')}</span>
        ${isConfirming
          ? `<button class="btn-ghost btn-sm" data-cancel-delete="${escapeHtml(p.name)}">Cancel</button>
             <button class="btn-pill red btn-sm" data-confirm-delete="${escapeHtml(p.name)}">${icon('trash')} Confirm</button>`
          : `<button class="btn-icon" data-ask-delete="${escapeHtml(p.name)}" title="Delete pipeline">${icon('trash')}</button>`
        }
        ${icon('chevronRight')}
      </div>
    </div>
  `;
}
