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
      // FastAPI's own validation errors (422 Unprocessable Entity — a
      // malformed request body, or, as happened once, a route decorator
      // accidentally landing on the wrong function and mismatching its
      // path-parameter type) return `detail` as an ARRAY of
      // {loc, msg, type} objects, not a string. Every other endpoint's
      // HTTPException(detail="...") in server.py sends a plain string,
      // which body.detail || detail happily used to just pass through —
      // but for the array case, that string wound up as ApiError.message
      // itself being an array, and this file's own runDetail.js template
      // literals turned that into the literal text "[object Object]"
      // wherever they interpolated err.message, instead of anything
      // useful. Join array entries into one readable string instead.
      if (Array.isArray(body.detail)) {
        detail = body.detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
      } else if (body.detail) {
        detail = body.detail;
      }
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
// Resumes a run that halted because a node's model call hit its output
// token cap (see server.py's POST /run/{run_uuid}/retry-truncated and
// pipeline/graph.py's _wrap_node_for_truncation_retry). Re-runs ONLY the
// node that truncated, with outputCap as its new max_tokens ceiling —
// nothing upstream of that node re-executes. Mirrors clarifyRun's
// resume-via-interrupt() mechanism; the two are separate endpoints
// (rather than one generic "resume" call) because the payloads differ
// in shape and 400 differently (clarifyRun always accepts free text,
// this one 400s on a non-positive cap).
export const retryTruncated = (runUuid, outputCap) =>
  request(`/run/${runUuid}/retry-truncated`, {
    method: 'POST', body: j({ output_cap: outputCap }),
  });
export const cancelRun = (runUuid) => request(`/run/${runUuid}`, { method: 'DELETE' });
export const streamUrl = (runUuid) => `${getApiBase()}/stream/${runUuid}`;
export const runImageUrl = (runUuid) => `${getApiBase()}/run/${runUuid}/image`;
// Excludes an attachment from future turns (kept on disk, still visible in
// artifacts) so it stops costing context tokens once it's no longer needed.
export const excludeAttachment = (runUuid, filename) =>
  request(`/run/${runUuid}/attachments/${encodeURIComponent(filename)}`, { method: 'DELETE' });
export const includeAttachment = (runUuid, filename) =>
  request(`/run/${runUuid}/attachments/${encodeURIComponent(filename)}/include`, { method: 'POST' });
// Adds new file(s) to an already-started run from runDetail.js's chat view
// (as opposed to runs.js's attachments, which only exist at run creation).
// attachments: [{ filename, content }], same shape as the POST /run payload.
// Picked up on the run's next turn — no need to wait for a terminal status.
export const addAttachments = (runUuid, attachments) =>
  request(`/run/${runUuid}/attachments`, { method: 'POST', body: j({ attachments }) });

// ── Chat ──────────────────────────────────────────────────────────────
// A chat is 1:1 with a run (chat_uuid == run_uuid). POST /run creates
// both. Use sendChatMessage only once the run has reached a terminal
// state (complete/unresolvable/error/cancelled/interrupted) — the
// backend 409s on an in-flight or clarification-paused run; route those
// through clarifyRun instead. See chat-ui-integration.md.
export const getChat = (runUuid) => request(`/chat/${runUuid}`);
// Deletes chat messages AND their per-turn artifact directories (the
// classify.json/draft.json/etc. snapshots shown in runDetail's advanced
// section). Does not touch the run itself or any distilled lessons.
// 409s if the run is still busy — same as sendChatMessage.
export const deleteChat = (runUuid) => request(`/chat/${runUuid}`, { method: 'DELETE' });
// replan=false (default): lighter follow-up, reuses this run's already-
// established classification/plan. replan=true: full re-plan from
// classify, as if task_type/profile could change based on the new message.
// Mirrors the explicit "Skip ensemble" toggle pattern in runs.js — this
// is a user-visible choice, not something inferred from message content.
//
// requestedProfile/task_type mirror runs.js's profile segmented control
// (see startRun) — only meaningful when replan=true, since a non-replan
// turn never reaches classify_node to apply them to. The server 400s if
// requestedProfile is set without replan=true. null (default) = auto,
// same convention as startRun.
//
// RENAMED from `mode` to `requestedProfile`: server.py's ChatMessageIn
// already has requested_profile (not mode) as the field it reads — mode
// is now informational-only on TaskClassification and no longer selects
// pipeline shape (see docs/pipeline-profile-escalation-design.md).
// "auto"/"short"/"medium"/"long"/"ultra" are the valid values now (an
// extra "medium" tier exists that didn't before), matching
// config/routing.yaml's pipeline_profiles.
//
// humanInTheLoop mirrors RunRequest.human_in_the_loop / ChatMessageIn.
// human_in_the_loop server-side — true (default) halts for clarification
// on low confidence once escalation is exhausted; false proceeds
// best-effort instead ("set-and-forget"). Like requestedProfile, only
// meaningful when replan=true.
//
// useSearch mirrors startRun's use_search toggle — was previously ABSENT
// from this call entirely (not just defaulted off), so a chat follow-up
// had no way to request search grounding at all, regardless of what the
// original run asked for. Unlike requestedProfile/task_type, this is NOT
// gated behind replan=true — the server applies it on every chat turn
// (see ChatMessageIn.use_search's docstring in server.py), so pass
// whatever state the person's search toggle is actually in for this
// message.
export const sendChatMessage = (
  runUuid, message, replan = false, requestedProfile = null, taskType = null,
  useSearch = false, humanInTheLoop = true,
) =>
  request(`/chat/${runUuid}`, {
    method: 'POST',
    body: j({
      message, replan,
      requested_profile:  requestedProfile,
      task_type:          taskType,
      use_search:         useSearch,
      human_in_the_loop:  humanInTheLoop,
    }),
  });

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
//
// ultra_plan/ultra_draft/ultra_appraise/ultra_critic REMOVED: these were
// dedicated roles that PIPELINE_ULTRA used to remap every stage onto
// (clients/llm.py's old _ULTRA_ROLE_MAP). Under the pipeline-profile
// design (docs/pipeline-profile-escalation-design.md), the "ultra"
// profile instead resolves each ordinary role (plan/draft/appraise/
// critic_a/critic_b/etc.) straight to its own escalation_ladders[role]
// final entry (config/models.yaml) — there's no separate ultra_* role to
// assign a model to anymore, so PATCH /models/{id}/role/ultra_plan (etc.)
// is no longer a valid target and shouldn't appear as an option here.
export const ALL_ROLES = [
  'vision_decode', 'classify', 'ideation', 'plan', 'draft', 'draft_short',
  'appraise', 'bugfix', 'critic_a', 'critic_b', 'synthesis_complex',
  'synthesis_simple', 'validate', 'final_validate', 'describe', 'distill',
  'gatekeeper', 'chess_fast', 'chess_slow',
];