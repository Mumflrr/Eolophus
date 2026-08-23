// ══════════════════════════════════════════════════════════════════════
// API client for the Eolophus Pipeline API (see api_contract.md).
// Every function here maps 1:1 to an endpoint. Swapping mock_server.py
// for server.py should require changing nothing in this file — that's
// the point of the contract being identical between the two.
// ══════════════════════════════════════════════════════════════════════

const DEFAULT_BASE = 'http://localhost:8000';
const STORAGE_KEY = 'eolophus.apiBase';

export function getApiBase() {
  return localStorage.getItem(STORAGE_KEY) || DEFAULT_BASE;
}

export function setApiBase(url) {
  const clean = url.trim().replace(/\/+$/, '');
  localStorage.setItem(STORAGE_KEY, clean || DEFAULT_BASE);
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  const base = getApiBase();
  let res;
  try {
    res = await fetch(`${base}${path}`, {
      headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
      ...options,
    });
  } catch (err) {
    throw new ApiError(`Can't reach the API at ${base} — is the server running?`, 0);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch { /* body wasn't JSON */ }
    throw new ApiError(detail, res.status);
  }
  if (res.status === 204) return null;
  return res.json();
}

const j = (body) => JSON.stringify(body);

// ── Runs ──────────────────────────────────────────────────────────────
export const startRun = (payload) => request('/run', { method: 'POST', body: j(payload) });
export const listRuns = (limit = 50) => request(`/runs?limit=${limit}`);
export const getRun = (runUuid) => request(`/run/${runUuid}`);
export const clarifyRun = (runUuid, answer) =>
  request(`/clarify/${runUuid}`, { method: 'POST', body: j({ answer }) });
export const cancelRun = (runUuid) => request(`/run/${runUuid}`, { method: 'DELETE' });
export const streamUrl = (runUuid) => `${getApiBase()}/stream/${runUuid}`;

// ── Models ────────────────────────────────────────────────────────────
export const listModels = () => request('/models');
export const loadModel = (modelId) => request(`/models/${modelId}/load`, { method: 'POST' });
export const unloadModel = (modelId) => request(`/models/${modelId}/unload`, { method: 'POST' });
export const reassignRole = (modelId, role) =>
  request(`/models/${modelId}/role/${role}`, { method: 'PATCH' });

// ── Config ────────────────────────────────────────────────────────────
export const getBudgets = () => request('/config/budgets');
export const patchBudgets = (budgets) =>
  request('/config/budgets', { method: 'PATCH', body: j({ budgets }) });
export const reloadConfig = () => request('/config/reload', { method: 'POST' });
export const getSearxng = () => request('/config/searxng');
export const setSearxng = (url) => request('/config/searxng', { method: 'POST', body: j({ url }) });

// ── Lessons ───────────────────────────────────────────────────────────
export const listLessons = (params = {}) => {
  const q = new URLSearchParams();
  if (params.task_type) q.set('task_type', params.task_type);
  if (params.issue_category) q.set('issue_category', params.issue_category);
  if (params.min_confidence) q.set('min_confidence', params.min_confidence);
  q.set('limit', params.limit || 200);
  return request(`/lessons?${q}`);
};
export const deleteLesson = (lessonUuid) => request(`/lessons/${lessonUuid}`, { method: 'DELETE' });
export const distillLessons = () => request('/lessons/distill', { method: 'POST' });

// ── Custom pipelines ─────────────────────────────────────────────────
// Wire shapes match server.py's PipelineDefIn/PipelineStepIn exactly (see
// custom_pipeline_handoff.md) — this doc's original contract file predates
// this surface, so these are documented here rather than in api_contract.md.
export const listPipelines = () => request('/pipelines');
export const getPipeline = (name) => request(`/pipelines/${encodeURIComponent(name)}`);
export const validatePipeline = (def) => request('/pipelines/validate', { method: 'POST', body: j(def) });
export const savePipeline = (def) => request('/pipelines', { method: 'POST', body: j(def) });
export const deletePipeline = (name) => request(`/pipelines/${encodeURIComponent(name)}`, { method: 'DELETE' });
export const getAvailableNodes = () => request('/pipelines/nodes/available');

// ── Health ────────────────────────────────────────────────────────────
export const getHealth = () => request('/health');

// All pipeline roles the backend recognizes (api_contract.md, PATCH /models/{id}/role/{role})
export const ALL_ROLES = [
  'vision_decode', 'classify', 'ideation', 'plan', 'draft', 'draft_short',
  'appraise', 'bugfix', 'critic_a', 'critic_b', 'synthesis_complex',
  'synthesis_simple', 'validate', 'final_validate', 'describe', 'distill',
  'gatekeeper', 'chess_fast', 'chess_slow', 'ultra_plan', 'ultra_draft',
  'ultra_appraise', 'ultra_critic',
];
