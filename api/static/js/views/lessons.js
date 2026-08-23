import { listLessons, deleteLesson, distillLessons, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { cardHeader, emptyState, loadingRow } from '../components/card.js';
import { escapeHtml, titleCase, fmtNumber, fmtRelativeTime } from '../format.js';
import { toastError, toastSuccess, toast } from '../toast.js';

let root = null;
let allLessons = [];
let fetchLimit = 200;
let confirmingDelete = null;

const filters = {
  taskType: '',
  issueCategory: '',
  tag: '',
  minConfidence: 0,
  sortBy: 'confidence_score',
};

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
          <h1>Lessons</h1>
          <div class="view-desc">Rules the pipeline has learned from past runs.</div>
        </div>
        <button class="btn-ghost" id="distill-btn">${icon('zap')} Distill lessons</button>
      </div>

      <section class="card">
        <div class="card-body no-header" style="padding-top:16px;">
          <div class="filter-bar" style="margin-bottom:12px;">
            <select class="select-input" id="filter-task-type" style="width:auto;min-width:150px;">
              <option value="">All task types</option>
            </select>
            <select class="select-input" id="filter-sort" style="width:auto;min-width:180px;">
              <option value="confidence_score">Sort: Confidence</option>
              <option value="times_retrieved">Sort: Times retrieved</option>
              <option value="times_useful">Sort: Times useful</option>
            </select>
            <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-secondary);">
              Min confidence
              <input type="number" class="number-input" id="filter-min-confidence" value="0" step="0.5" min="0" style="width:64px;">
            </label>
          </div>
          <div class="filter-bar" id="category-chips"></div>
          <div style="height:8px"></div>
          <div class="filter-bar" id="tag-chips"></div>
        </div>
      </section>

      <section class="card">
        ${cardHeader({ icon: 'book', color: 'purple', title: 'Learned rules', actionHtml: `<span class="text-tertiary mono" id="lesson-count" style="font-size:11.5px;"></span>` })}
        <div class="card-body no-header" id="lesson-list">${loadingRow('Loading lessons…')}</div>
        <div style="padding:0 18px 16px;" id="load-more-wrap"></div>
      </section>
    </div>
  `;

  document.getElementById('filter-task-type').addEventListener('change', (e) => { filters.taskType = e.target.value; renderList(); });
  document.getElementById('filter-sort').addEventListener('change', (e) => { filters.sortBy = e.target.value; renderList(); });
  document.getElementById('filter-min-confidence').addEventListener('input', (e) => {
    filters.minConfidence = Number(e.target.value) || 0;
    renderList();
  });
  document.getElementById('distill-btn').addEventListener('click', onDistill);
}

async function load(more = false) {
  const listEl = document.getElementById('lesson-list');
  try {
    const lessons = await listLessons({ limit: fetchLimit });
    allLessons = lessons;
    if (!root) return;
    populateFilterOptions();
    renderList();
  } catch (err) {
    if (!root || !listEl) return;
    listEl.innerHTML = emptyState({
      iconName: 'alertTriangle',
      title: "Can't load lessons",
      sub: err instanceof ApiError ? err.message : 'Check the API base URL in Settings.',
    });
  }
}

function populateFilterOptions() {
  const taskTypes = [...new Set(allLessons.map((l) => l.task_type).filter(Boolean))].sort();
  const sel = document.getElementById('filter-task-type');
  const current = filters.taskType;
  sel.innerHTML = `<option value="">All task types</option>` +
    taskTypes.map((t) => `<option value="${t}" ${current === t ? 'selected' : ''}>${escapeHtml(titleCase(t))}</option>`).join('');

  const categories = [...new Set(allLessons.map((l) => l.issue_category).filter(Boolean))].sort();
  document.getElementById('category-chips').innerHTML = categories.map((c) => `
    <button class="chip ${filters.issueCategory === c ? 'active' : ''}" data-cat="${c}">${escapeHtml(titleCase(c))}</button>
  `).join('');
  document.querySelectorAll('[data-cat]').forEach((chip) => {
    chip.addEventListener('click', () => {
      filters.issueCategory = filters.issueCategory === chip.dataset.cat ? '' : chip.dataset.cat;
      populateFilterOptions();
      renderList();
    });
  });

  const tags = [...new Set(allLessons.flatMap((l) => l.tags || []))].sort();
  document.getElementById('tag-chips').innerHTML = tags.map((t) => `
    <button class="chip ${filters.tag === t ? 'active' : ''}" data-tag="${t}">#${escapeHtml(t)}</button>
  `).join('');
  document.querySelectorAll('[data-tag]').forEach((chip) => {
    chip.addEventListener('click', () => {
      filters.tag = filters.tag === chip.dataset.tag ? '' : chip.dataset.tag;
      populateFilterOptions();
      renderList();
    });
  });
}

function visibleLessons() {
  let out = allLessons.filter((l) =>
    (!filters.taskType || l.task_type === filters.taskType) &&
    (!filters.issueCategory || l.issue_category === filters.issueCategory) &&
    (!filters.tag || (l.tags || []).includes(filters.tag)) &&
    (l.confidence_score >= filters.minConfidence)
  );
  out.sort((a, b) => (b[filters.sortBy] || 0) - (a[filters.sortBy] || 0));
  return out;
}

function renderList() {
  const listEl = document.getElementById('lesson-list');
  const countEl = document.getElementById('lesson-count');
  const visible = visibleLessons();
  if (countEl) countEl.textContent = `${visible.length} of ${allLessons.length}`;

  if (!visible.length) {
    listEl.innerHTML = emptyState({
      iconName: 'book',
      title: allLessons.length ? 'No lessons match these filters' : 'No lessons yet',
      sub: allLessons.length ? 'Try clearing a filter.' : 'The pipeline will record lessons as it corrects mistakes across runs.',
    });
    document.getElementById('load-more-wrap').innerHTML = '';
    return;
  }

  listEl.innerHTML = visible.map(lessonCard).join('');
  wireLessonActions();

  const moreWrap = document.getElementById('load-more-wrap');
  moreWrap.innerHTML = allLessons.length >= fetchLimit
    ? `<button class="btn-ghost" id="load-more-btn">Load more</button>`
    : '';
  const moreBtn = document.getElementById('load-more-btn');
  if (moreBtn) moreBtn.addEventListener('click', () => { fetchLimit += 200; load(); });
}

function lessonCard(l) {
  const lowSignal = l.times_retrieved > 5 && l.times_useful === 0;
  const isConfirming = confirmingDelete === l.lesson_uuid;
  return `
    <div class="card lesson-card">
      <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
        <div class="tag-row">
          <span class="tag">${escapeHtml(titleCase(l.issue_category))}</span>
          <span class="tag">${escapeHtml(titleCase(l.task_type))}</span>
          ${(l.tags || []).map((t) => `<span class="tag">#${escapeHtml(t)}</span>`).join('')}
          ${lowSignal ? `<span class="badge orange">low signal</span>` : ''}
        </div>
        <span class="badge purple" style="flex-shrink:0;">conf ${l.confidence_score.toFixed(1)}</span>
      </div>
      <div class="lesson-resolution">${escapeHtml(l.resolution_pattern)}</div>
      ${l.issue_summary ? `<div class="lesson-summary">${escapeHtml(l.issue_summary)}</div>` : ''}
      <div class="lesson-stats">
        <span><strong>${fmtNumber(l.times_seen)}</strong> seen</span>
        <span><strong>${fmtNumber(l.times_retrieved)}</strong> retrieved</span>
        <span class="${lowSignal ? 'low-signal' : ''}"><strong>${fmtNumber(l.times_useful)}</strong> useful</span>
        <span>${escapeHtml(l.model_caught || '—')}</span>
        <span>${fmtRelativeTime(l.last_triggered)}</span>
      </div>
      <div style="margin-top:12px;display:flex;justify-content:flex-end;gap:8px;">
        ${isConfirming
          ? `<span class="text-tertiary" style="font-size:12px;align-self:center;">Delete this lesson?</span>
             <button class="btn-pill grey btn-sm" data-cancel-delete="${l.lesson_uuid}">Cancel</button>
             <button class="btn-pill red btn-sm" data-confirm-delete="${l.lesson_uuid}">${icon('trash')} Confirm</button>`
          : `<button class="btn-ghost btn-sm" data-ask-delete="${l.lesson_uuid}">${icon('trash')} Delete</button>`
        }
      </div>
    </div>
  `;
}

function wireLessonActions() {
  document.querySelectorAll('[data-ask-delete]').forEach((btn) => {
    btn.addEventListener('click', () => { confirmingDelete = btn.dataset.askDelete; renderList(); });
  });
  document.querySelectorAll('[data-cancel-delete]').forEach((btn) => {
    btn.addEventListener('click', () => { confirmingDelete = null; renderList(); });
  });
  document.querySelectorAll('[data-confirm-delete]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const uuid = btn.dataset.confirmDelete;
      btn.disabled = true;
      try {
        await deleteLesson(uuid);
        allLessons = allLessons.filter((l) => l.lesson_uuid !== uuid);
        confirmingDelete = null;
        toastSuccess('Lesson deleted.');
        renderList();
      } catch (err) {
        toastError(err instanceof ApiError ? err.message : 'Could not delete lesson.');
        btn.disabled = false;
      }
    });
  });
}

async function onDistill() {
  const btn = document.getElementById('distill-btn');
  btn.disabled = true;
  try {
    const res = await distillLessons();
    toast(res.message || 'Distillation requested.', res.status === 'not_implemented' ? 'info' : 'success');
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Distillation request failed.');
  } finally {
    btn.disabled = false;
  }
}
