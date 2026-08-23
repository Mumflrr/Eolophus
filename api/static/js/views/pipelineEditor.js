import { getPipeline, savePipeline, validatePipeline, getAvailableNodes, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { escapeHtml, titleCase } from '../format.js';
import { toastError, toastSuccess } from '../toast.js';
import { computeLayout, computeEdges, edgePath, NODE_W, NODE_H } from '../pipelineLayout.js';

let root = null;
let def = null;          // working definition, wire-shaped + local-only _x/_y overrides
let isNew = false;
let originalName = null; // to detect a rename (create-under-new-name vs overwrite)
let selectedStepId = null;
let availableNodes = null; // cached /pipelines/nodes/available response
let dragOverride = new Map(); // stepId -> {x,y} session-only manual drag position
let validation = { valid: true, errors: [] };
let validateTimer = null;
let saving = false;
let addMenuOpen = false;
let pendingLoopCap = null; // {stepId, outcomeIndex} awaiting a max_iterations value

const BLANK_DEF = () => ({
  name: '', description: '', entry_step: '', steps: [], edge_overrides: {}, max_total_iterations: 20,
});

export async function mount(el, { name }) {
  root = el;
  isNew = !name;
  originalName = name;
  selectedStepId = null;
  dragOverride = new Map();
  validation = { valid: true, errors: [] };
  saving = false;
  addMenuOpen = false;
  pendingLoopCap = null;

  renderShell();

  try {
    availableNodes = availableNodes || await getAvailableNodes();
  } catch (err) {
    toastError('Could not load available node types — add-step options may be incomplete.');
    availableNodes = { existing_nodes: [], step_types: {} };
  }

  if (isNew) {
    def = BLANK_DEF();
    renderAll();
  } else {
    try {
      def = await getPipeline(name);
      normalizeDef();
      renderAll();
      scheduleValidate();
    } catch (err) {
      document.getElementById('pipeline-canvas-wrap').innerHTML = `
        <div class="inspector-empty" style="height:100%;">
          ${icon('alertTriangle')}
          <div>${escapeHtml(err instanceof ApiError ? err.message : "Couldn't load this pipeline.")}</div>
        </div>`;
    }
  }
}

export function unmount() {
  clearTimeout(validateTimer);
  root = null;
  def = null;
}

// Wire-format defs from the server may omit fields our editor always
// wants present (edge_overrides, outcomes arrays) — normalize once on load.
function normalizeDef() {
  def.edge_overrides = def.edge_overrides || {};
  def.max_total_iterations = def.max_total_iterations ?? 20;
  for (const s of def.steps) {
    if (s.type === 'decision') s.outcomes = s.outcomes || [];
  }
}

function render() { renderAll(); }

function renderShell() {
  root.innerHTML = `
    <div class="view-inner" style="gap:14px;">
      <div class="pipeline-toolbar">
        <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0;">
          <button class="btn-ghost" id="back-to-list" title="All pipelines">${icon('arrowLeft')}</button>
          <input type="text" class="pipeline-name-input" id="pipeline-name-input" placeholder="pipeline_name" value="">
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <div style="position:relative;">
            <button class="btn-ghost" id="add-step-btn">${icon('plus')} Add step</button>
            <div id="add-step-menu-mount"></div>
          </div>
          <button class="btn btn-primary" id="save-pipeline-btn" disabled>${icon('check')} Save</button>
        </div>
      </div>
      <input type="text" class="text-input" id="pipeline-desc-input" placeholder="Description (optional)" style="max-width:520px;">
      <div class="pipeline-editor-shell">
        <div class="pipeline-canvas-wrap" id="pipeline-canvas-wrap">
          <div class="loading-row" style="padding:20px;"><span class="spinner"></span>Loading…</div>
        </div>
        <aside class="pipeline-inspector" id="pipeline-inspector"></aside>
      </div>
    </div>
  `;
  document.getElementById('back-to-list').addEventListener('click', () => { location.hash = '#/pipelines'; });
  document.getElementById('add-step-btn').addEventListener('click', toggleAddMenu);
}

function renderAll() {
  if (!root || !def) return;
  document.getElementById('pipeline-name-input').value = def.name;
  document.getElementById('pipeline-desc-input').value = def.description || '';
  renderCanvas();
  renderInspector();
  updateSaveState();
  wireToolbarInputs();
}

function wireToolbarInputs() {
  const nameInput = document.getElementById('pipeline-name-input');
  const descInput = document.getElementById('pipeline-desc-input');
  nameInput.oninput = () => { def.name = nameInput.value; updateSaveState(); scheduleValidate(); };
  descInput.oninput = () => { def.description = descInput.value; updateSaveState(); };
  document.getElementById('save-pipeline-btn').onclick = onSave;
}

// ── Canvas ────────────────────────────────────────────────────────────

function effectivePosition(stepId, layoutPositions) {
  if (dragOverride.has(stepId)) return dragOverride.get(stepId);
  const p = layoutPositions.get(stepId);
  return p ? { x: p.x, y: p.y } : { x: 0, y: 0 };
}

function renderCanvas() {
  const wrap = document.getElementById('pipeline-canvas-wrap');
  if (!wrap) return;

  if (!def.steps.length) {
    wrap.innerHTML = `
      <div class="inspector-empty" style="height:100%;">
        ${icon('git')}
        <div>No steps yet</div>
        <div style="max-width:260px;">Use "Add step" above to start with an existing node, a freeform prompt, or a decision branch.</div>
      </div>`;
    return;
  }

  const { positions, width, height } = computeLayout(def);
  const edges = computeEdges(def, positions);

  // Track how many loop-back edges land on each (from,to) pair so we can
  // offset their curves and avoid exact overlap.
  const loopEdgeIndex = new Map();

  const edgeSvgParts = [];
  for (const e of edges) {
    if (!e.to) continue; // __end__ edges aren't drawn to a node — see the end marker below
    const fromPos = effectivePosition(e.from, positions);
    const toPos = effectivePosition(e.to, positions);
    let idx = 0;
    if (e.isLoopBack) {
      const key = `${e.from}->${e.to}`;
      idx = loopEdgeIndex.get(key) || 0;
      loopEdgeIndex.set(key, idx + 1);
    }
    const d = edgePath(fromPos, toPos, e.isLoopBack, idx);
    const midX = (fromPos.x + NODE_W + toPos.x) / 2;
    const midY = e.isLoopBack
      ? Math.max(fromPos.y, toPos.y) + NODE_H + 50 + idx * 22
      : (fromPos.y + toPos.y) / 2 + NODE_H / 2 - 6;
    edgeSvgParts.push(`<path class="edge-path${e.isLoopBack ? ' loop-back' : ''}" d="${d}"></path>`);
    if (e.label) {
      edgeSvgParts.push(`<text class="edge-label${e.isLoopBack ? ' loop-back-label' : ''}" x="${midX}" y="${midY}" text-anchor="middle">${escapeHtml(e.label)}</text>`);
    }
  }

  const nodeHtml = def.steps.map((step) => {
    const pos = effectivePosition(step.id, positions);
    const meta = positions.get(step.id);
    const unreached = meta ? meta.unreached : true;
    return stepNodeHtml(step, pos, unreached);
  }).join('');

  wrap.innerHTML = `
    <div class="pipeline-canvas" style="width:${width}px;height:${height + 60}px;">
      <svg class="pipeline-edges-svg" width="${width}" height="${height + 60}">
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--c-blue)" opacity="0.7"></path>
          </marker>
          <marker id="arrowhead-orange" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--c-orange)" opacity="0.8"></path>
          </marker>
        </defs>
        ${edgeSvgParts.join('')}
      </svg>
      ${nodeHtml}
    </div>
  `;

  wireCanvasInteractions(positions);
}

function wireCanvasInteractions(positions) {
  const wrap = document.getElementById('pipeline-canvas-wrap');

  wrap.querySelectorAll('[data-step-node]').forEach((nodeEl) => {
    const stepId = nodeEl.dataset.stepNode;

    nodeEl.addEventListener('click', (e) => {
      if (e.target.closest('[data-remove-step]')) return;
      selectedStepId = stepId;
      renderCanvas();
      renderInspector();
    });

    let dragging = false;
    let startMouse = { x: 0, y: 0 };
    let startPos = { x: 0, y: 0 };

    nodeEl.addEventListener('mousedown', (e) => {
      if (e.target.closest('[data-remove-step]')) return;
      dragging = true;
      startMouse = { x: e.clientX, y: e.clientY };
      const current = effectivePosition(stepId, positions);
      startPos = { ...current };
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - startMouse.x;
      const dy = e.clientY - startMouse.y;
      dragOverride.set(stepId, { x: Math.max(0, startPos.x + dx), y: Math.max(0, startPos.y + dy) });
      renderCanvas();
    });
    document.addEventListener('mouseup', () => { dragging = false; });
  });

  wrap.querySelectorAll('[data-remove-step]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeStep(btn.dataset.removeStep);
    });
  });
}

function removeStep(stepId) {
  def.steps = def.steps.filter((s) => s.id !== stepId);
  // Clean up anything that referenced it — dangling references are still
  // caught by the validator, but auto-cleaning obvious cases (this step's
  // own edge_override, outcomes pointing at it) keeps the graph tidy
  // without the user having to hunt down every reference by hand.
  delete def.edge_overrides[stepId];
  for (const [k, v] of Object.entries(def.edge_overrides)) {
    if (v === stepId) delete def.edge_overrides[k];
  }
  for (const s of def.steps) {
    if (s.type === 'decision') {
      s.outcomes = (s.outcomes || []).filter((o) => o.next_step !== stepId);
    }
  }
  if (def.entry_step === stepId) def.entry_step = def.steps[0]?.id || '';
  if (selectedStepId === stepId) selectedStepId = null;
  dragOverride.delete(stepId);
  renderAll();
  scheduleValidate();
}

function stepNodeHtml(step, pos, unreached) {
  const typeClass = `node-${step.type}`;
  const isEntry = step.id === def.entry_step;
  const isSelected = step.id === selectedStepId;
  let sub = '';
  if (step.type === 'existing') sub = step.node_name || '(no node selected)';
  else if (step.type === 'freeform') sub = step.model ? `model: ${step.model}` : '(no model set)';
  else if (step.type === 'decision') sub = `${(step.outcomes || []).length} outcome${(step.outcomes || []).length === 1 ? '' : 's'}`;

  return `
    <div class="pipeline-node ${typeClass}${isSelected ? ' selected' : ''}${unreached ? ' unreached' : ''}"
         data-step-node="${escapeHtml(step.id)}"
         style="left:${pos.x}px;top:${pos.y}px;">
      <button class="pipeline-node-remove" data-remove-step="${escapeHtml(step.id)}" title="Delete step">${icon('x')}</button>
      <div class="pipeline-node-head">
        <span class="pipeline-node-type-tag ${typeClass}">${step.type}</span>
        ${isEntry ? `<span class="pipeline-node-entry-badge">${icon('play', '')} entry</span>` : ''}
      </div>
      <div class="pipeline-node-id">${escapeHtml(step.id)}</div>
      <div class="pipeline-node-sub">${escapeHtml(sub)}</div>
    </div>
  `;
}

// ── Add-step menu ────────────────────────────────────────────────────

function toggleAddMenu() {
  addMenuOpen = !addMenuOpen;
  renderAddMenu();
}

function renderAddMenu() {
  const mount_ = document.getElementById('add-step-menu-mount');
  if (!mount_) return;
  if (!addMenuOpen) { mount_.innerHTML = ''; return; }
  mount_.innerHTML = `
    <div class="add-step-menu">
      <button data-add-type="existing">${icon('cpu', '')} Existing node</button>
      <button data-add-type="freeform">${icon('edit', '')} Freeform prompt</button>
      <button data-add-type="decision">${icon('git', '')} Decision branch</button>
    </div>
  `;
  mount_.querySelectorAll('[data-add-type]').forEach((btn) => {
    btn.addEventListener('click', () => {
      addStep(btn.dataset.addType);
      addMenuOpen = false;
      renderAddMenu();
    });
  });
  // Close on outside click.
  setTimeout(() => {
    document.addEventListener('click', closeAddMenuOnOutsideClick, { once: true });
  }, 0);
}

function closeAddMenuOnOutsideClick(e) {
  if (e.target.closest('#add-step-btn') || e.target.closest('.add-step-menu')) {
    // re-arm listener since the menu might still be open (e.g. clicked the toggle itself)
    if (addMenuOpen) document.addEventListener('click', closeAddMenuOnOutsideClick, { once: true });
    return;
  }
  addMenuOpen = false;
  renderAddMenu();
}

function uniqueStepId(prefix) {
  const existing = new Set(def.steps.map((s) => s.id));
  let i = 1;
  let id = `${prefix}_${i}`;
  while (existing.has(id)) { i++; id = `${prefix}_${i}`; }
  return id;
}

function addStep(type) {
  let step;
  if (type === 'existing') {
    step = { type: 'existing', id: uniqueStepId('step'), node_name: (availableNodes.existing_nodes || [])[0] || '' };
  } else if (type === 'freeform') {
    step = {
      type: 'freeform', id: uniqueStepId('step'), model: '9b', budget_tokens: 0, thinking: false,
      system_prompt: '', user_template: '{input}', input_key: 'normalised_input', output_key: uniqueStepId('output'),
      feedback_mode: 'auto',
    };
  } else {
    step = {
      type: 'decision', id: uniqueStepId('decision'), model: '9b', thinking: false, budget_tokens: 0,
      system_prompt: '', input_key: 'normalised_input',
      outcomes: [{ value: 'pass', next_step: '__end__' }, { value: 'retry', next_step: '__end__' }],
      is_loop_back: false, max_iterations: null, feedback_mode: 'auto',
    };
  }
  def.steps.push(step);
  if (!def.entry_step) def.entry_step = step.id;
  selectedStepId = step.id;
  renderAll();
  scheduleValidate();
}

// ── Inspector panel ──────────────────────────────────────────────────

function renderInspector() {
  const panel = document.getElementById('pipeline-inspector');
  if (!panel) return;
  const step = def.steps.find((s) => s.id === selectedStepId);

  const validationHtml = renderValidationPanel();

  if (!step) {
    panel.innerHTML = `
      <div class="inspector-empty">
        ${icon('layers')}
        <div>Select a step to edit it, or add a new one above.</div>
      </div>
      ${validationHtml}
    `;
    return;
  }

  panel.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:12px;flex:1;min-height:0;overflow-y:auto;">
      ${stepInspectorFields(step)}
    </div>
    ${validationHtml}
  `;
  wireInspectorFields(step);
}

function stepInspectorFields(step) {
  const idField = `
    <div class="inspector-field">
      <label class="field-label">Step ID</label>
      <input type="text" class="text-input" id="insp-id" value="${escapeHtml(step.id)}">
    </div>
    <div class="inspector-field">
      <label class="field-label" style="display:flex;align-items:center;justify-content:space-between;">
        Entry point
        <span style="font-weight:400;color:var(--text-tertiary);">${step.id === def.entry_step ? 'this step' : ''}</span>
      </label>
      <button class="btn-ghost btn-sm" id="insp-set-entry" ${step.id === def.entry_step ? 'disabled' : ''} style="width:100%;justify-content:center;">
        ${step.id === def.entry_step ? 'This is the entry step' : 'Make this the entry step'}
      </button>
    </div>
  `;

  if (step.type === 'existing') {
    const nodeOptions = (availableNodes.existing_nodes || []);
    return idField + `
      <div class="inspector-field">
        <label class="field-label">Existing node</label>
        <select class="select-input" id="insp-node-name">
          ${nodeOptions.map((n) => `<option value="${n}" ${step.node_name === n ? 'selected' : ''}>${escapeHtml(titleCase(n))}</option>`).join('')}
        </select>
      </div>
      <div class="inspector-field">
        <label class="field-label">Model override <span style="font-weight:400;color:var(--text-tertiary);">(optional)</span></label>
        <input type="text" class="text-input" id="insp-model-override" placeholder="leave blank to use configured role default" value="${escapeHtml(step.model_override || '')}">
      </div>
      <div class="inspector-field">
        <label class="field-label">Budget override <span style="font-weight:400;color:var(--text-tertiary);">(optional, -1 = unlimited)</span></label>
        <input type="number" class="number-input" id="insp-budget-override" style="width:100%;text-align:left;" min="-1" placeholder="unset" value="${step.budget_override ?? ''}">
      </div>
    `;
  }

  if (step.type === 'freeform') {
    return idField + `
      <div class="inspector-field">
        <label class="field-label">Model</label>
        <input type="text" class="text-input" id="insp-model" value="${escapeHtml(step.model || '')}" placeholder="e.g. 9b">
      </div>
      <div class="inspector-field">
        <label class="field-label">Budget tokens <span style="font-weight:400;color:var(--text-tertiary);">(-1 unlimited, 0 no thinking)</span></label>
        <input type="number" class="number-input" id="insp-budget-tokens" style="width:100%;text-align:left;" min="-1" value="${step.budget_tokens ?? 0}">
      </div>
      <div class="inspector-field">
        <div class="feedback-toggle-row">
          <span>Thinking enabled</span>
          <button type="button" class="toggle ${step.thinking ? 'on' : ''}" id="insp-thinking"></button>
        </div>
      </div>
      <div class="inspector-field">
        <label class="field-label">System prompt</label>
        <textarea class="textarea-input" id="insp-system-prompt" placeholder="Full system prompt for this node">${escapeHtml(step.system_prompt || '')}</textarea>
      </div>
      <div class="inspector-field">
        <label class="field-label">User template</label>
        <textarea class="textarea-input" id="insp-user-template" placeholder="{input}">${escapeHtml(step.user_template || '{input}')}</textarea>
      </div>
      <div class="inspector-field">
        <label class="field-label">Input key</label>
        <input type="text" class="text-input" id="insp-input-key" value="${escapeHtml(step.input_key || 'normalised_input')}">
      </div>
      <div class="inspector-field">
        <label class="field-label">Output key</label>
        <input type="text" class="text-input" id="insp-output-key" value="${escapeHtml(step.output_key || '')}">
      </div>
      ${feedbackModeField(step)}
    `;
  }

  // decision
  return idField + `
      <div class="inspector-field">
        <label class="field-label">Model <span style="font-weight:400;color:var(--text-tertiary);">(defaults to 9b — decisions are cheap)</span></label>
        <input type="text" class="text-input" id="insp-model" value="${escapeHtml(step.model || '9b')}">
      </div>
      <div class="inspector-field">
        <label class="field-label">System prompt</label>
        <textarea class="textarea-input" id="insp-system-prompt" placeholder="What to evaluate and how to decide">${escapeHtml(step.system_prompt || '')}</textarea>
      </div>
      <div class="inspector-field">
        <label class="field-label">Input key</label>
        <input type="text" class="text-input" id="insp-input-key" value="${escapeHtml(step.input_key || 'normalised_input')}">
      </div>
      <div class="inspector-field">
        <label class="field-label" style="display:flex;justify-content:space-between;">
          Outcomes
          <button class="btn-icon" id="insp-add-outcome" title="Add outcome" style="width:20px;height:20px;">${icon('plus')}</button>
        </label>
        <div id="insp-outcomes">${outcomesHtml(step)}</div>
      </div>
      ${loopCapField(step)}
      ${feedbackModeField(step)}
  `;
}

function outcomesHtml(step) {
  const stepOptions = def.steps.filter((s) => s.id !== step.id).map((s) => s.id);
  return (step.outcomes || []).map((o, i) => `
    <div class="outcome-row" data-outcome-index="${i}">
      <input type="text" class="text-input" data-outcome-field="value" value="${escapeHtml(o.value)}" placeholder="value">
      <select class="select-input" data-outcome-field="next_step">
        <option value="__end__" ${o.next_step === '__end__' ? 'selected' : ''}>→ end</option>
        ${stepOptions.map((sid) => `<option value="${escapeHtml(sid)}" ${o.next_step === sid ? 'selected' : ''}>→ ${escapeHtml(sid)}</option>`).join('')}
      </select>
      <button class="btn-icon" data-remove-outcome="${i}" style="width:26px;height:26px;flex-shrink:0;" title="Remove outcome">${icon('x')}</button>
    </div>
  `).join('');
}

function loopCapField(step) {
  return `
    <div class="inspector-field">
      <div class="feedback-toggle-row">
        <span>Loop-back (this decision can cycle)</span>
        <button type="button" class="toggle ${step.is_loop_back ? 'on' : ''}" id="insp-loop-back"></button>
      </div>
      ${step.is_loop_back ? `
        <div style="margin-top:8px;">
          <label class="field-label">Max iterations <span style="font-weight:400;color:var(--text-tertiary);">(required for a loop-back)</span></label>
          <input type="number" class="number-input" id="insp-max-iterations" style="width:100%;text-align:left;" min="1" value="${step.max_iterations ?? ''}" placeholder="e.g. 3">
        </div>
      ` : ''}
    </div>
    ${pendingLoopCap && pendingLoopCap.stepId === step.id ? `
      <div class="loop-cap-prompt">
        ${icon('alertTriangle')} <strong>This edge routes backward — that's a loop.</strong>
        Set a cap on "${escapeHtml(step.id)}" before it can be saved.
      </div>
    ` : ''}
  `;
}

function feedbackModeField(step) {
  return `
    <div class="inspector-field">
      <div class="feedback-toggle-row">
        <span>Pass feedback forward <span style="color:var(--text-tertiary);">(prior decision's reasoning as {feedback})</span></span>
        <button type="button" class="toggle ${step.feedback_mode !== 'none' ? 'on' : ''}" id="insp-feedback-mode"></button>
      </div>
    </div>
  `;
}

function renderValidationPanel() {
  if (!validation) return '';
  if (validation.pending) {
    return `<div class="validation-panel"><div class="loading-row" style="padding:0;"><span class="spinner"></span>Validating…</div></div>`;
  }
  if (validation.valid) {
    return `<div class="validation-panel"><div class="validation-ok">${icon('checkCircle')} Valid — ready to save</div></div>`;
  }
  return `
    <div class="validation-panel">
      <div class="validation-errors">
        ${validation.errors.map((e) => `<div class="validation-error-item">${icon('alertTriangle')}<span>${escapeHtml(e)}</span></div>`).join('')}
      </div>
    </div>
  `;
}

function wireInspectorFields(step) {
  const byId = (id) => document.getElementById(id);
  const onChange = () => { renderAll(); scheduleValidate(); };

  byId('insp-id').addEventListener('change', (e) => {
    const newId = e.target.value.trim();
    if (!newId || newId === step.id) { e.target.value = step.id; return; }
    if (def.steps.some((s) => s.id === newId)) { toastError(`Step id '${newId}' is already in use.`); e.target.value = step.id; return; }
    renameStepId(step.id, newId);
    selectedStepId = newId;
    onChange();
  });

  byId('insp-set-entry').addEventListener('click', () => { def.entry_step = step.id; onChange(); });

  if (step.type === 'existing') {
    byId('insp-node-name').addEventListener('change', (e) => { step.node_name = e.target.value; onChange(); });
    byId('insp-model-override').addEventListener('input', (e) => { step.model_override = e.target.value.trim() || null; scheduleValidate(); });
    byId('insp-budget-override').addEventListener('input', (e) => {
      const v = e.target.value.trim();
      step.budget_override = v === '' ? null : Number(v);
      scheduleValidate();
    });
  } else if (step.type === 'freeform') {
    byId('insp-model').addEventListener('input', (e) => { step.model = e.target.value; scheduleValidate(); });
    byId('insp-budget-tokens').addEventListener('input', (e) => { step.budget_tokens = Number(e.target.value) || 0; scheduleValidate(); });
    byId('insp-thinking').addEventListener('click', (e) => { step.thinking = !step.thinking; e.target.classList.toggle('on', step.thinking); scheduleValidate(); });
    byId('insp-system-prompt').addEventListener('input', (e) => { step.system_prompt = e.target.value; scheduleValidate(); });
    byId('insp-user-template').addEventListener('input', (e) => { step.user_template = e.target.value; scheduleValidate(); });
    byId('insp-input-key').addEventListener('change', (e) => { step.input_key = e.target.value.trim(); onChange(); });
    byId('insp-output-key').addEventListener('change', (e) => { step.output_key = e.target.value.trim(); onChange(); });
    wireFeedbackToggle(step);
  } else {
    byId('insp-model').addEventListener('input', (e) => { step.model = e.target.value; scheduleValidate(); });
    byId('insp-system-prompt').addEventListener('input', (e) => { step.system_prompt = e.target.value; scheduleValidate(); });
    byId('insp-input-key').addEventListener('change', (e) => { step.input_key = e.target.value.trim(); onChange(); });
    byId('insp-add-outcome').addEventListener('click', () => {
      step.outcomes = step.outcomes || [];
      step.outcomes.push({ value: `outcome_${step.outcomes.length + 1}`, next_step: '__end__' });
      onChange();
    });
    wireOutcomeRows(step);
    wireLoopBackToggle(step);
    wireFeedbackToggle(step);
  }
}

function wireOutcomeRows(step) {
  document.querySelectorAll('#insp-outcomes [data-outcome-index]').forEach((row) => {
    const idx = Number(row.dataset.outcomeIndex);
    row.querySelector('[data-outcome-field="value"]').addEventListener('input', (e) => {
      step.outcomes[idx].value = e.target.value;
      scheduleValidate();
    });
    const nextSelect = row.querySelector('[data-outcome-field="next_step"]');
    nextSelect.addEventListener('change', (e) => {
      const target = e.target.value;
      step.outcomes[idx].next_step = target;
      checkLoopBackNeeded(step, target);
      renderAll();
      scheduleValidate();
    });
  });
  document.querySelectorAll('[data-remove-outcome]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const idx = Number(btn.dataset.removeOutcome);
      step.outcomes.splice(idx, 1);
      renderAll();
      scheduleValidate();
    });
  });
}

// Mirrors custom_validator.py's _check_undeclared_cycles: a decision
// outcome routing to a step at or before its own position is a cycle. If
// the user just wired that and hasn't marked is_loop_back, surface the
// prompt immediately rather than waiting for a save-time rejection —
// this is the specific interaction the handoff doc asked for.
function checkLoopBackNeeded(step, targetId) {
  if (targetId === '__end__') return;
  const order = def.steps.map((s) => s.id);
  const selfIdx = order.indexOf(step.id);
  const targetIdx = order.indexOf(targetId);
  const isBackward = targetIdx !== -1 && targetIdx <= selfIdx;
  if (isBackward && !step.is_loop_back) {
    pendingLoopCap = { stepId: step.id };
  } else if (!isBackward && pendingLoopCap && pendingLoopCap.stepId === step.id) {
    pendingLoopCap = null;
  }
}

function wireLoopBackToggle(step) {
  document.getElementById('insp-loop-back').addEventListener('click', (e) => {
    step.is_loop_back = !step.is_loop_back;
    if (!step.is_loop_back) step.max_iterations = null;
    else pendingLoopCap = null;
    renderAll();
    scheduleValidate();
  });
  const capInput = document.getElementById('insp-max-iterations');
  if (capInput) {
    capInput.addEventListener('input', (e) => {
      const v = Number(e.target.value);
      step.max_iterations = v > 0 ? v : null;
      if (step.max_iterations) pendingLoopCap = null;
      scheduleValidate();
    });
  }
}

function wireFeedbackToggle(step) {
  document.getElementById('insp-feedback-mode').addEventListener('click', (e) => {
    step.feedback_mode = step.feedback_mode === 'none' ? 'auto' : 'none';
    e.target.classList.toggle('on', step.feedback_mode !== 'none');
    scheduleValidate();
  });
}

function renameStepId(oldId, newId) {
  const step = def.steps.find((s) => s.id === oldId);
  step.id = newId;
  if (def.entry_step === oldId) def.entry_step = newId;
  if (def.edge_overrides[oldId] !== undefined) {
    def.edge_overrides[newId] = def.edge_overrides[oldId];
    delete def.edge_overrides[oldId];
  }
  for (const k of Object.keys(def.edge_overrides)) {
    if (def.edge_overrides[k] === oldId) def.edge_overrides[k] = newId;
  }
  for (const s of def.steps) {
    if (s.type === 'decision') {
      for (const o of (s.outcomes || [])) {
        if (o.next_step === oldId) o.next_step = newId;
      }
    }
  }
  if (dragOverride.has(oldId)) {
    dragOverride.set(newId, dragOverride.get(oldId));
    dragOverride.delete(oldId);
  }
}

// ── Validation (debounced, live) ────────────────────────────────────

function scheduleValidate() {
  clearTimeout(validateTimer);
  validation = { ...validation, pending: true };
  renderValidationPanelInPlace();
  validateTimer = setTimeout(runValidate, 500);
}

function renderValidationPanelInPlace() {
  const panel = document.getElementById('pipeline-inspector');
  if (!panel) return;
  const existing = panel.querySelector('.validation-panel');
  if (existing) existing.outerHTML = renderValidationPanel();
}

async function runValidate() {
  if (!def || !def.name || !def.entry_step || !def.steps.length) {
    validation = { valid: false, errors: ['Set a name, an entry step, and at least one step before validating.'], pending: false };
    renderValidationPanelInPlace();
    updateSaveState();
    return;
  }
  try {
    const res = await validatePipeline(toWirePayload());
    validation = { valid: res.valid, errors: res.errors || [], pending: false };
  } catch (err) {
    validation = { valid: false, errors: [err instanceof ApiError ? err.message : 'Validation request failed.'], pending: false };
  }
  renderValidationPanelInPlace();
  updateSaveState();
}

function toWirePayload() {
  // Strip local-only fields (none currently live on `def` itself — drag
  // positions are kept entirely outside it in dragOverride) and coerce
  // numeric fields that inputs may have left as strings.
  return JSON.parse(JSON.stringify(def));
}

function updateSaveState() {
  const btn = document.getElementById('save-pipeline-btn');
  if (!btn) return;
  const hasBasics = !!(def && def.name && def.entry_step && def.steps.length);
  btn.disabled = saving || !hasBasics;
}

async function onSave() {
  if (!def.name) { toastError('Give the pipeline a name first.'); return; }
  if (pendingLoopCap) { toastError('Set a max iterations cap on the pending loop-back before saving.'); return; }
  saving = true;
  const btn = document.getElementById('save-pipeline-btn');
  btn.disabled = true;
  const label = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> Saving…`;
  try {
    await savePipeline(toWirePayload());
    toastSuccess(`Saved '${def.name}'.`);
    isNew = false;
    originalName = def.name;
    location.hash = `#/pipelines/${encodeURIComponent(def.name)}`;
  } catch (err) {
    if (err instanceof ApiError && err.status === 400) {
      toastError('Pipeline failed validation — see the panel for details.');
      scheduleValidate();
    } else {
      toastError(err instanceof ApiError ? err.message : 'Could not save pipeline.');
    }
  } finally {
    saving = false;
    btn.innerHTML = label;
    updateSaveState();
  }
}