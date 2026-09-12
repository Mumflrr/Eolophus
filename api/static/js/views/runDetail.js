import { getRun, getChat, sendChatMessage, clarifyRun, retryTruncated, cancelRun, streamUrl, excludeAttachment, includeAttachment, addAttachments, ApiError } from '../api.js';
import { icon } from '../icons.js';
import { emptyState, loadingRow } from '../components/card.js';
import { fmtMs, fmtNumber, fmtRelativeTime, titleCase, escapeHtml, prettyJson, statusColor } from '../format.js';
import { toastError, toastSuccess } from '../toast.js';
import { setUltraAmbient } from '../ambient.js';

// ── Minimal markdown renderer for chat/assistant message content ──────────
// Deliberately hand-rolled rather than pulling in an npm markdown library:
// this project's module-loading setup (bundler vs native ES modules served
// directly to the browser) wasn't confirmed, so a bare-specifier import
// (e.g. `import { marked } from 'marked'`) risked silently failing to
// resolve depending on that setup. If a bundler with npm access IS
// available, swapping this out for marked/markdown-it later only touches
// renderMarkdown()'s internals — every call site below stays the same.
//
// SAFETY: raw text is escaped via escapeHtml() FIRST, then markdown syntax
// is applied to the now-safe, already-escaped string. This means nothing
// in the source text — including text an LLM generated, which is not
// fully trusted input — can produce live HTML by containing something
// that looks like a tag; the only HTML tags that ever appear in the
// output are the ones this function explicitly writes (<strong>, <em>,
// <code>, <a>, <ul>/<li>, <h1-3>, <br>). Link hrefs are also constrained
// to http(s)/mailto — no javascript: URIs.
//
// Intentionally NOT supported: raw inline HTML passthrough, images,
// tables, nested blockquotes, footnotes. This covers what model output
// and chat replies actually tend to use (paragraphs, emphasis, inline
// and fenced code, links, simple lists, headers) without the complexity
// (and larger escaping surface) of a full CommonMark implementation.
function renderMarkdown(raw) {
  if (!raw) return '';

  // Fenced code blocks first, pulled out and replaced with placeholders —
  // their content must NOT have inline markdown (bold/italic/links)
  // applied inside it, and doing this before escaping the rest avoids
  // double-escaping the code itself.
  const codeBlocks = [];
  let text = String(raw).replace(/```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre class="chat-code-block"><code${lang ? ` class="lang-${escapeHtml(lang)}"` : ''}>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`);
    return `\u0000CODEBLOCK${idx}\u0000`;
  });

  // Escape everything else now — all subsequent replacements operate on
  // already-safe text and only ever ADD the specific tags below.
  text = escapeHtml(text);

  // Inline code spans — also excluded from further inline formatting.
  const codeSpans = [];
  text = text.replace(/`([^`\n]+)`/g, (_, code) => {
    const idx = codeSpans.length;
    codeSpans.push(`<code class="chat-inline-code">${code}</code>`);
    return `\u0000CODESPAN${idx}\u0000`;
  });

  // Headers (#, ##, ###) — line-anchored, ATX-style only.
  text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  text = text.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold and italic — bold (**/__) before italic (*/_) so **x** doesn't
  // get partially consumed by the italic pass first.
  text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/__([^_\n]+)__/g, '<strong>$1</strong>');
  text = text.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  text = text.replace(/(?<![A-Za-z0-9])_([^_\n]+)_(?![A-Za-z0-9])/g, '<em>$1</em>');

  // Links: [text](url) — only http(s)/mailto schemes make it through;
  // anything else (e.g. javascript:) renders as plain escaped text
  // instead of a link, since escapeHtml() already ran on the URL too.
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  // Simple unordered lists: consecutive lines starting with "- " or "* ".
  text = text.replace(/(?:^[-*] .+$\n?)+/gm, (block) => {
    const items = block.trim().split('\n').map((line) => `<li>${line.replace(/^[-*] /, '')}</li>`).join('');
    return `<ul>${items}</ul>\n`;
  });

  // Remaining single newlines -> <br>, so plain paragraphs still wrap
  // visually without requiring the model to emit <p> or blank-line pairs.
  text = text.replace(/\n/g, '<br>');

  // Restore code spans and blocks last, so their contents (which may
  // themselves contain characters that look like the patterns above,
  // e.g. asterisks in code) were never subject to the passes above.
  text = text.replace(/\u0000CODESPAN(\d+)\u0000/g, (_, i) => codeSpans[Number(i)]);
  text = text.replace(/\u0000CODEBLOCK(\d+)\u0000/g, (_, i) => codeBlocks[Number(i)]);

  return text;
}

// Copies raw text (NOT rendered HTML — the original markdown-source
// message content, so pasting elsewhere gives clean text/markdown rather
// than a wall of <br>/<strong> tags) to the clipboard, with a brief
// visual confirmation on the trigger button.
//
// navigator.clipboard.writeText requires a secure context (HTTPS or
// localhost) — falls back to the older execCommand('copy') approach via
// a hidden textarea for plain HTTP deployments, since this project's
// deployment context wasn't confirmed to always be HTTPS/localhost.
async function copyToClipboard(text, btn) {
  const showCopied = () => {
    if (!btn) return;
    const original = btn.textContent;
    btn.textContent = '✓';
    btn.disabled = true;
    setTimeout(() => {
      btn.textContent = original;
      btn.disabled = false;
    }, 1200);
  };

  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      showCopied();
      return;
    }
  } catch {
    // fall through to the manual fallback below
  }

  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    showCopied();
  } catch {
    toastError('Could not copy to clipboard.');
  }
}

let root = null;
let es = null;
let runUuid = null;
let data = { status: null, stages: [], artifacts: {}, iterations: {}, chatArtifacts: {}, turnIterations: {}, clarification: null, truncation: null, mode: null, profile: null };
let clarifySubmitting = false;
let truncationRetrySubmitting = false;
let attachmentActionPending = null; // filename currently being excluded/included, or null
let attachmentUploadPending = false; // true while a newly-picked file is being read/sent

// Chat state — a chat is 1:1 with a run (chat_uuid == run_uuid). See
// chat-ui-integration.md. `messages` is oldest-first, exactly as the API
// returns it. `drawerExpanded` controls whether we're showing just the
// last two turns or the full scrollable history.
let chat = { messages: [], loaded: false };
let drawerExpanded = false;
let chatSending = false;
let openDetailSeq = null; // seq of the message whose detail panel is open, or null
let chatReplan = false;   // explicit toggle — see api.js sendChatMessage comment
// Per-turn profile override, mirroring runs.js's profile segmented
// control. '' = auto. Only sent when chatReplan is on — see the
// profile-picker's display:none gating in renderChat() and the server's
// 400 otherwise. Renamed from chatMode — see CHAT_PROFILES's comment.
let chatProfile = '';
// Explicit toggle for search grounding on chat follow-ups — mirrors
// runs.js's use_search checkbox on the initial run. UNLIKE chatProfile,
// this is NOT gated behind chatReplan: the server applies use_search to
// every chat turn regardless of replan (see ChatMessageIn.use_search's
// docstring in server.py — plan_node/ideation_node read it directly off
// whatever state _run_chat_replan builds, and that's the only state-
// construction path a chat turn goes through today). This was
// previously not exposed anywhere in the chat UI at all — sendChatMessage
// had no useSearch argument, so a follow-up question had no way to
// request search regardless of what was toggled on the original run.
let chatUseSearch = false;
// Per-turn human-in-the-loop override, mirroring runs.js's toggle (added
// alongside it — see runs.js's toggle-human-in-the-loop). true (default,
// matches ChatMessageIn.human_in_the_loop's server-side default): low
// confidence still eventually halts for clarification once a stage's
// escalation ladder is exhausted. false ("set-and-forget"): proceed
// best-effort instead of halting. Like chatProfile, only meaningful when
// chatReplan is on — a non-replan turn never reaches classify_node, so
// there's nothing for this to apply to.
let chatHumanInTheLoop = true;
let lessonPollTimer = null;
let chatLoadSeq = 0;      // guards against out-of-order loadChat() responses —
                           // see loadChat() below
let lastChatFingerprint = null; // last chat.messages payload we actually rendered —
                           // see renderChat(); skips the rebuild when the 5s lesson
                           // poller comes back with nothing new, which otherwise
                           // replayed the turn fade-in animation every poll and
                           // read as a visible "flash"
const LESSON_POLL_MS = 5000; // per-chat, faster than the run-level 15s pattern
                              // in models.js — lessons can land mid-turn, not
                              // just at run boundaries

const TERMINAL_STATUSES = ['complete', 'unresolvable', 'error', 'cancelled', 'interrupted'];

// Mirrors runs.js's attachment constants — keep in sync with each other
// and with MAX_ATTACHMENT_CHARS server-side.
const MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024; // 2MB per file
const ACCEPTED_EXTENSIONS = '.py,.md,.txt,.json,.js,.jsx,.ts,.tsx,.yaml,.yml,.sh,.csv,.html,.css,.toml,.ini,.log,.rs,.go,.java,.c,.cpp,.h,.rb,.sql';

// Mirrors runs.js's PROFILES exactly — same options, same meaning.
// Kept as its own copy rather than importing from runs.js since runs.js
// isn't a shared module (it also owns page-local state like `state`).
//
// RENAMED from CHAT_MODES/chatMode under the pipeline-profile design (see
// docs/pipeline-profile-escalation-design.md and server.py's RunRequest/
// ChatMessageIn, which send requested_profile, not mode — mode is now
// informational-only on TaskClassification and no longer drives routing).
// Added "medium" as a real option (previously only two tiers existed
// here: short/long). api.js's sendChatMessage() has been updated to
// match — its former `mode` argument is now `requestedProfile`, sent as
// requested_profile, with a new humanInTheLoop argument alongside it
// (see the chat-human-in-the-loop-row toggle below).
const CHAT_PROFILES = [
  { id: '', label: 'Auto' },
  { id: 'short', label: 'Short' },
  { id: 'medium', label: 'Medium' },
  { id: 'long', label: 'Long' },
  { id: 'ultra', label: 'Ultra' },
];

const ARTIFACT_ORDER = [
  'classification', 'planspec', 'draft', 'appraisal_report', 'audit', 'fixed',
  'critique', 'verdict', 'final', 'final_validation', 'run',
];

// Which run artifact (if any) best represents each node's work, for the
// chat-bubble detail view keyed by a message's node_id (describe/clarify/
// truncated — see server.py's append_message call sites). `plan` is the
// only one with a node_id that ALSO shows up as a chat message today, so
// it's the only entry here; gatekeeper now writes audit.json too (see
// nodes/gatekeeper.py) but never posts its own chat message, so there's
// no node_id="gatekeeper" bubble for this map to apply to in practice —
// its artifact still surfaces fine through the top-level/per-turn
// ARTIFACT_ORDER list in renderArtifacts(), just not via this map.
const NODE_ARTIFACT_KEY = {
  plan: 'planspec',
};

export function mount(el, { runUuid: uuid }) {
  root = el;
  runUuid = uuid;
  data = { status: null, stages: [], artifacts: {}, iterations: {}, chatArtifacts: {}, turnIterations: {}, clarification: null, truncation: null, mode: null, profile: null };
  chat = { messages: [], loaded: false };
  drawerExpanded = false;
  chatSending = false;
  openDetailSeq = null;
  chatReplan = false;
  chatUseSearch = false;
  chatLoadSeq = 0; // any in-flight loadChat() from a previous run is now stale
  lastChatFingerprint = null; // force the first renderChat() for this run to draw
  drawerTransitioning = false;
  drawerJustExpandedAt = 0;
  openArtifacts = new Set();
  selectedIteration = {};
  attachmentActionPending = null;
  attachmentUploadPending = false;
  renderSkeleton();
  loadInitial();
  startLessonPolling();
}

export function unmount() {
  closeStream();
  stopLessonPolling();
  setUltraAmbient(false);
  root = null;
  runUuid = null;
}

// Lessons can be written mid-turn (distiller_node fires as part of the
// same graph invocation that produces the reply) and used-lessons are
// only known once retrieval happens server-side — polling GET /chat
// (which now includes lessons_used per message, see api contract) is
// simpler and more robust than trying to thread a third event type
// through the existing stage SSE stream. Runs the whole time the run
// view is mounted, not just while the drawer is expanded, since a
// collapsed drawer still shows the last two turns and their lesson tags.
function startLessonPolling() {
  stopLessonPolling();
  lessonPollTimer = setInterval(() => {
    if (!root || !runUuid) return;
    loadChat();
  }, LESSON_POLL_MS);
}

function stopLessonPolling() {
  clearInterval(lessonPollTimer);
  lessonPollTimer = null;
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
    await loadChat();
  } catch (err) {
    if (!root) return;
    document.getElementById('run-body').innerHTML = emptyState({
      iconName: 'alertTriangle',
      title: "Couldn't load this run",
      sub: err instanceof ApiError ? err.message : 'It may not exist, or the API base URL is wrong.',
    });
  }
}

async function loadChat() {
  // loadChat() is called from ~5 independent triggers (initial load, the
  // 5s lesson poller, both SSE sentinel handlers, and clarify/chat submit)
  // and nothing serialized them — whichever fetch happened to RESOLVE
  // last won and overwrote chat.messages, regardless of which call was
  // actually the most recent request. With rapid clarify rounds this
  // showed up as the drawer staying one turn behind: an earlier, slower
  // request would land after a newer, faster one and stomp its result.
  // Stamp each call with an increasing sequence number and drop any
  // response that isn't from the most recent call by the time it resolves.
  const mySeq = ++chatLoadSeq;
  try {
    const c = await getChat(runUuid);
    if (mySeq !== chatLoadSeq) return; // a newer loadChat() superseded this one
    chat.messages = c.messages || [];
    chat.loaded = true;
    renderChat();
  } catch {
    if (mySeq !== chatLoadSeq) return;
    // A run created before chat existed may 404 here — treat as an empty
    // chat rather than an error state; the rest of the page still works.
    chat.loaded = true;
    renderChat();
  }
}

function applyRunDetail(r) {
  data.status = r.status;
  data.stages = r.stages || [];
  data.artifacts = r.artifacts || {};
  // Per-turn artifact snapshots — {seq: {classification: {...}, draft: {...}, ...}}.
  // See server.py's GET /run/{run_uuid}: every chat turn writes into its
  // own run_dir/turns/<seq>/ instead of overwriting the top-level files
  // in data.artifacts, which is why data.artifacts alone only ever
  // reflects the very first (non-chat) run — chat replies need this
  // separate map to show their own artifacts. Keys arrive as strings
  // (JSON object keys / directory names); message.seq is a number, so
  // callers must String(seq) when looking this up.
  data.chatArtifacts = r.chat_artifacts || {};
  // Per-iteration snapshots for loop-writable artifacts (fixed/verdict/
  // draft/critique — see write_iteration_artifact in clients/llm.py) —
  // {iteration: {fixed: {...}, verdict: {...}, ...}}. Empty {} for a
  // single-pass run that never looped (no iterations/ dir was ever
  // written), which is the common case and renders identically to
  // before this existed. turnIterations mirrors this per chat-turn seq,
  // the same way chatArtifacts mirrors artifacts.
  data.iterations     = r.iterations || {};
  data.turnIterations = r.turn_iterations || {};

  // justAnsweredClarification guards against re-showing the clarification
  // box after the user has already answered it client-side (see
  // onClarifySubmit). Without this, ANY call to applyRunDetail() between
  // "user hit Answer" and "server actually reports a non-waiting status"
  // — a slow SSE reconnect via openStream()'s onerror handler, a manual
  // refresh, a periodic poll, anything that re-fetches GET /run/{uuid} —
  // would blindly copy r.clarification back into data.clarification if
  // the server's response still reflects the pre-answer state (e.g. a
  // clarification.json sentinel not yet cleared server-side at the exact
  // moment of that particular request — the same class of race as the
  // cancel/delete sentinel bug fixed in server.py's remove_run). The
  // symptom was the answered clarification box silently reappearing with
  // no further action from the user.
  //
  // The flag is set the instant the user submits (onClarifySubmit) and
  // is only cleared once the server confirms a genuinely NEW state that
  // isn't "still waiting for clarification" — at that point any FUTURE
  // clarification (a later turn in the same chat) is a new, real one and
  // should display normally again.
  if (data.justAnsweredClarification && r.status === 'waiting_for_clarification') {
    // Stale read — server hasn't caught up yet. Keep showing "answered".
    data.clarification = null;
  } else {
    data.justAnsweredClarification = false;
    data.clarification = r.clarification || null;
  }
  data.truncation = r.truncation || null;

  if (r.artifacts && r.artifacts.run) {
    data.mode = r.artifacts.run.mode || data.mode;
    // r.artifacts.run.profile is the resolved pipeline_profiles name
    // (routing.yaml) — "short"|"medium"|"long"|"ultra" — set by
    // classify_node (see pipeline/routers.py's select_profile /
    // resolve_profile). Tracked separately from data.mode: mode is
    // TaskClassification.mode, whose Mode enum only ever contains
    // "short"/"long" — it was NEVER capable of being "ultra", so the
    // data.mode === 'ultra' check below was dead code even before the
    // profile design existed. profile is the actual source of truth now.
    data.profile = r.artifacts.run.profile || data.profile;
  }
  setUltraAmbient(data.profile === 'ultra' || isUltraRun());
}

function isUltraRun() {
  // model_swap stage-log fallback kept as a heuristic for runs from
  // before the run artifact carried a profile field at all (or any
  // future case where profile resolution didn't happen for some
  // reason) — profile==='ultra' above is the real signal now.
  return data.stages.some((s) => s.stage === 'model_swap') || data.profile === 'ultra';
}

function openStream() {
  closeStream();
  const isTerminal = TERMINAL_STATUSES.includes(data.status);
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
      renderChat(); // input-gating depends on status
      closeStream();
      loadChat(); // pick up the clarifying question as a chat turn too
      return;
    }
    if (payload.type === 'truncated') {
      data.status = 'waiting_for_truncation_retry';
      data.truncation = {
        stage:          payload.stage,
        node:           payload.node,
        cap:            payload.cap,
        tokens_out:     payload.tokens_out,
        thinking_block: payload.thinking_block,
        partial_answer: payload.partial_answer,
      };
      renderStatusBadge();
      renderTruncation();
      renderTimeline();
      renderChat(); // input-gating depends on status
      closeStream();
      loadChat(); // pick up the "what it was thinking" message as a chat turn too
      return;
    }
    if (payload.type === 'complete') {
      data.status = payload.status;
      closeStream();
      renderStatusBadge();
      renderTimeline();
      renderChat();
      refreshArtifacts();
      loadChat(); // new assistant turn(s) landed in chats.db, not on this stream
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
    renderStatusBadge();
    renderTimeline();
    renderArtifacts();
    renderAttachments();
    renderClarification();
    renderTruncation();
    renderChat();
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
    <div id="truncate-mount"></div>
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

    <section class="card" id="artifacts-section" style="display:none;">
      <div class="card-header" style="padding-bottom:6px;">
        <div class="card-icon-badge purple">${icon('book')}</div>
        <div class="card-header-text">
          <div class="card-title">Artifacts</div>
          <div class="card-subtitle">Raw JSON written to disk by this run — classification, plan, draft, and so on</div>
        </div>
      </div>
      <div class="card-body no-header" id="artifacts-mount"></div>
    </section>

    <section class="card" id="attachments-section">
      <div class="card-header" style="padding-bottom:6px;">
        <div class="card-icon-badge blue">${icon('paperclip')}</div>
        <div class="card-header-text">
          <div class="card-title">Attachments</div>
          <div class="card-subtitle">Files shared with this run. Remove one to stop it being included in future messages — it stays visible here either way.</div>
        </div>
      </div>
      <div class="card-body no-header">
        <div id="attachments-mount"></div>
        <label class="btn-ghost" for="attachment-add-input" id="attachment-add-label" style="width:fit-content;cursor:pointer;margin-top:8px;">
          ${icon('paperclip')} Add files
        </label>
        <input type="file" id="attachment-add-input" accept="${ACCEPTED_EXTENSIONS}" multiple style="display:none;">
        <div class="text-tertiary" style="font-size:11px;margin-top:4px;">
          Text and code files only — added to this run's context from your next message onward.
        </div>
      </div>
    </section>

    <div id="chat-drawer" class="chat-drawer">
      <button class="chat-drawer-handle" id="chat-drawer-handle" title="Expand conversation">
        <span class="chat-drawer-handle-label">
          ${icon('chevronUp', 'chat-drawer-chevron')}
          Conversation
        </span>
      </button>
      <div class="chat-drawer-body" id="chat-drawer-body"></div>
      <div class="chat-replan-row" title="Off: quick follow-up, reuses this run's existing plan. On: full re-plan from scratch, as if task type/profile could change.">
        <span class="row-title">Re-plan next message</span>
        <button type="button" class="toggle" id="toggle-replan" aria-pressed="false" aria-label="Full re-plan on next message"></button>
      </div>
      <div class="chat-replan-row" title="Fold web search results into this message's prompt. Applies regardless of re-plan — plan_node/ideation_node read this directly.">
        <span class="row-title">Search the web</span>
        <button type="button" class="toggle${chatUseSearch ? ' on' : ''}" id="toggle-search" aria-pressed="${chatUseSearch}" aria-label="Use web search for next message"></button>
      </div>
      <div class="chat-profile-row" id="chat-profile-row" style="display:${chatReplan ? '' : 'none'};" title="Only applies when re-plan is on — a quick follow-up never re-classifies, so there's nothing to pin this to.">
        <span class="row-title">Profile</span>
        <div class="segmented" id="chat-profile-segmented" role="group" aria-label="Profile">
          ${CHAT_PROFILES.map((m) => `<button type="button" data-chat-profile="${m.id}" class="${m.id === 'ultra' ? 'danger' : ''} ${chatProfile === m.id ? 'active' : ''}">${m.label}</button>`).join('')}
        </div>
      </div>
      <div class="chat-replan-row" id="chat-human-in-the-loop-row" style="display:${chatReplan ? '' : 'none'};" title="Only applies when re-plan is on. On (default): pause and ask if classify/plan run out of ways to improve confidence on their own. Off: proceed with the best available result instead of stopping to ask.">
        <span class="row-title">Ask if unsure</span>
        <button type="button" class="toggle${chatHumanInTheLoop ? ' on' : ''}" id="toggle-human-in-the-loop" aria-pressed="${chatHumanInTheLoop}" aria-label="Halt for clarification when confidence stays low"></button>
      </div>
      <form class="chat-input-row" id="chat-input-form">
        <textarea class="textarea-input" id="chat-input" placeholder="Message…" rows="1"></textarea>
        <button type="submit" class="btn-icon" id="chat-send-btn" title="Send">${icon('send')}</button>
      </form>
    </div>
  `;
  renderStatusBadge();
  renderTimeline();
  renderArtifacts();
  renderAttachments();
  renderClarification();
  renderTruncation();
  renderChat();

  document.getElementById('chat-drawer-handle').addEventListener('click', () => {
    drawerExpanded = !drawerExpanded;
    renderChat();
  });
  const body = document.getElementById('chat-drawer-body');
  body.addEventListener('wheel', onDrawerWheel, { passive: false });
  document.getElementById('chat-input-form').addEventListener('submit', onChatSubmit);
  document.getElementById('attachment-add-input').addEventListener('change', onAttachmentAdd);
  document.getElementById('toggle-replan').addEventListener('click', (e) => {
    chatReplan = !chatReplan;
    e.currentTarget.classList.toggle('on', chatReplan);
    e.currentTarget.setAttribute('aria-pressed', String(chatReplan));
    if (!chatReplan) chatProfile = ''; // reset — profile only makes sense alongside replan
    const profileRow = document.getElementById('chat-profile-row');
    if (profileRow) profileRow.style.display = chatReplan ? '' : 'none';
    const hitlRow = document.getElementById('chat-human-in-the-loop-row');
    if (hitlRow) hitlRow.style.display = chatReplan ? '' : 'none';
    renderChatProfilePicker();
  });
  document.getElementById('toggle-search').addEventListener('click', (e) => {
    chatUseSearch = !chatUseSearch;
    e.currentTarget.classList.toggle('on', chatUseSearch);
    e.currentTarget.setAttribute('aria-pressed', String(chatUseSearch));
  });
  document.getElementById('toggle-human-in-the-loop').addEventListener('click', (e) => {
    chatHumanInTheLoop = !chatHumanInTheLoop;
    e.currentTarget.classList.toggle('on', chatHumanInTheLoop);
    e.currentTarget.setAttribute('aria-pressed', String(chatHumanInTheLoop));
  });
  document.getElementById('chat-profile-segmented').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-chat-profile]');
    if (!btn) return;
    chatProfile = btn.dataset.chatProfile;
    document.querySelectorAll('#chat-profile-segmented button')
      .forEach((b) => b.classList.toggle('active', b === btn));
  });
}

// Re-renders just the active state of the profile segmented buttons —
// avoids a full renderChat() (which would replay message fade-ins, see
// lastChatFingerprint) just to reflect the replan-toggle's reset of
// chatProfile back to ''.
function renderChatProfilePicker() {
  document.querySelectorAll('#chat-profile-segmented button[data-chat-profile]').forEach((b) => {
    b.classList.toggle('active', b.dataset.chatProfile === chatProfile);
  });
}

// Scrolling up while over the (collapsed) drawer expands it; scrolling
// down while already at the top of the expanded, fully-scrolled-up
// drawer collapses it back. Once expanded, normal scroll behavior inside
// the body takes over for paging through history.
//
// A single trackpad/wheel gesture fires many onwheel events in quick
// succession. Without a guard, each one re-set drawerExpanded and called
// renderChat() again mid-transition, restarting/interrupting the
// max-height animation and reading as a flash rather than one glide.
// drawerTransitioning debounces that: once we've triggered a toggle, we
// ignore further wheel-driven toggles until the CSS transition finishes.
let drawerTransitioning = false;

// Matches the 420ms max-height transition in chat-drawer.css, plus a
// small buffer so a slow frame doesn't let a new toggle sneak in before
// the CSS transition has actually finished settling.
const DRAWER_TRANSITION_MS = 420 + 60;

function beginDrawerTransition() {
  drawerTransitioning = true;
  setTimeout(() => { drawerTransitioning = false; }, DRAWER_TRANSITION_MS);
}

// Scrolling up while over the (collapsed) drawer expands it; scrolling
// down while already at the top of the expanded, fully-scrolled-up
// drawer collapses it back. Once expanded, normal scroll behavior inside
// the body takes over for paging through history.
//
// Bug this guards against: scrollTop is 0 both (a) right after expanding,
// before the user has scrolled at all, and (b) after the user has
// scrolled down through history and back up to the top. Those two cases
// look identical to a "scrollTop <= 0" check, so the very first downward
// wheel tick right after expanding was being read as "at top, scrolling
// down => collapse", collapsing the drawer before the user could ever
// move scrollTop off zero. drawerJustExpandedAt tracks a short settle
// window right after expansion during which downward scrolling is always
// treated as normal scrolling, never as a collapse trigger.
let drawerJustExpandedAt = 0;
const DRAWER_SETTLE_MS = 500;

function onDrawerWheel(e) {
  const body = e.currentTarget;

  // This element owns its own scroll entirely — never let a wheel event
  // over it fall through to scrolling the page. overscroll-behavior:
  // contain (in chat-drawer.css) covers the case where the body has
  // scrollable content and hits its own top/bottom; this preventDefault
  // is the backstop for the collapsed state, where the body may have no
  // internal scroll room at all for "contain" to hook into. Since this
  // blocks the browser's native wheel-scroll too, we apply the delta to
  // scrollTop ourselves for the normal in-bounds case below.
  e.preventDefault();

  if (drawerTransitioning) return;

  if (!drawerExpanded && e.deltaY < 0) {
    drawerExpanded = true;
    drawerJustExpandedAt = performance.now();
    beginDrawerTransition();
    renderChat();
    return;
  }

  const justExpanded = performance.now() - drawerJustExpandedAt < DRAWER_SETTLE_MS;
  if (drawerExpanded && e.deltaY > 0 && body.scrollTop <= 0 && !justExpanded) {
    drawerExpanded = false;
    beginDrawerTransition();
    renderChat();
    return;
  }

  body.scrollTop += e.deltaY;
}

function renderStatusBadge() {
  const sub = document.getElementById('run-subtitle');
  const actions = document.getElementById('run-header-actions');
  if (!sub || !actions) return;
  const color = statusColor(data.status);
  sub.innerHTML = `<span class="badge ${color}">${escapeHtml(titleCase(data.status || 'unknown'))}</span>`;

  const cancellable = ['running', 'pending', 'waiting_for_clarification', 'waiting_for_truncation_retry'].includes(data.status);
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

  mount_.innerHTML = runTotalsSummary() + nodes.join('') + trailing;
}

// Sums tokens_in/tokens_out across every stage entry, and separately
// finds the largest single memory_mib.gpu_total seen (not a sum — each
// entry is a snapshot of ONE model's footprint at its own load moment,
// and those models are flash-swapped in and out of the same 10GB card
// one at a time per model_manager.py's EXCLUSIVE_MODELS/_stop_current,
// never resident simultaneously, so adding them together would imply
// concurrent usage that never actually happens; the peak single load is
// the number worth knowing).
function runTotalsSummary() {
  let tokensIn = 0, tokensOut = 0, peakGpuMib = null;
  for (const s of data.stages) {
    tokensIn  += s.tokens_in  || 0;
    tokensOut += s.tokens_out || 0;
    const g = s.memory_mib && s.memory_mib.gpu_total;
    if (g != null && (peakGpuMib === null || g > peakGpuMib)) peakGpuMib = g;
  }
  if (!tokensIn && !tokensOut && peakGpuMib === null) return '';

  const peakLabel = peakGpuMib === null
    ? null
    : (peakGpuMib >= 1024 ? `${(peakGpuMib / 1024).toFixed(2)} GiB` : `${Math.round(peakGpuMib)} MiB`);

  return `
    <div class="run-totals-summary">
      <span class="meta-item"><strong>${fmtNumber(tokensIn)}</strong> in / <strong>${fmtNumber(tokensOut)}</strong> out total</span>
      ${peakLabel ? `<span class="meta-item" title="Largest single model load seen this run (models are flash-swapped, not concurrent — see runDetail.js comment)">peak ${peakLabel}</span>` : ''}
    </div>
  `;
}

function stageNode(s, prev, isLast) {
  const isSwapEntry   = s.stage === 'model_swap';
  // Synthetic marker written by POST /clarify right when a clarification
  // round resumes — see server.py's clarify_run. Distinguishes "3 classify
  // entries because 3 separate resumed rounds happened" from "3 classify
  // entries because something retried silently", which the timeline
  // previously had no way to show.
  const isResumeEntry = s.stage === 'clarification_resumed';
  const dotClass = (isSwapEntry || isResumeEntry) ? 'active' : (s.status === 'ok' || !s.status) ? '' : 'error';
  const showSwap = !isSwapEntry && !isResumeEntry && prev && prev.model && s.model && prev.model !== s.model;
  // An escalation (design doc §2.3/§2.7): call_role's internal escalation
  // walk (clients/llm.py) re-dispatches to the NEXT model on the same
  // stage's ladder after a truncation or low-confidence result, logging a
  // second stages.log entry for the identical `stage` value with a
  // different `model`. That's distinguishable from an ordinary pipeline
  // transition to a new role (e.g. classify → draft), which also changes
  // `model` between consecutive entries but changes `stage` too — hence
  // checking prev.stage === s.stage specifically, not just prev.model !==
  // s.model (which showSwap above already covers for the flash-swap
  // case). Purely a display distinction — the underlying stages.log
  // schema/fields are unchanged; this reads the same data two different
  // consecutive entries already carry.
  const isEscalation = showSwap && prev.stage === s.stage;

  if (isResumeEntry) {
    return `
      <div class="stage-node">
        <div class="stage-node-rail">
          <span class="stage-node-dot ${dotClass}"></span>
          ${isLast ? '' : '<span class="stage-node-line"></span>'}
        </div>
        <div class="stage-node-body">
          <div class="stage-swap">${icon('zap')} resumed with clarification answer</div>
        </div>
      </div>
    `;
  }

  const thinkBar = s.think_ratio > 0 ? `
    <span class="think-bar" title="Fraction of the response spent thinking">
      <span class="think-bar-track"><span class="think-bar-fill" style="width:${Math.round(s.think_ratio * 100)}%"></span></span>
      <span>${Math.round(s.think_ratio * 100)}% think</span>
    </span>` : '';

  const retryBadge = s.retries > 0 ? `<span class="badge orange">${s.retries} retr${s.retries === 1 ? 'y' : 'ies'}</span>` : '';
  const statusBadge = (s.status && s.status !== 'ok') ? `<span class="badge red">${escapeHtml(s.status)}</span>` : '';

  // memory_mib is only present on the (rare) stage entry where a model
  // load actually just happened — see llm.py's did_load check. Most
  // stages have no memory_mib at all, which is correct: it's a load
  // event, not a per-call measurement. gpu_total may itself be null even
  // when memory_mib is present, if -lv 4 wasn't set for that model's
  // launch script (see model_memory.py's docstring) — guard both levels.
  //
  // NOTE: formatted inline (not via a fmtMib()-style helper) because
  // format.js's actual exports weren't available to check against when
  // this was written — confirm whether format.js already has a
  // MiB/GiB-aware formatter and swap this for that if so, rather than
  // this being a second, slightly-different formatting convention living
  // alongside it. Same reasoning for not using icon('cpu') here — that
  // key's existence in icons.js wasn't verified, and a wrong key could
  // silently render nothing depending on how icon() handles misses.
  // .stage-escalation needs a CSS rule added wherever .stage-swap's own
  // rule lives — that's a run-detail-specific stylesheet, not
  // components.css (checked: components.css has the shared card/badge/
  // toggle primitives, but no .stage-swap/.stage-node/.stage-escalation
  // classes at all, so this timeline's styling is defined elsewhere).
  // components.css DOES already establish --c-purple/--c-purple-dim/
  // --c-purple-border as this app's convention for a highlighted/active
  // accent (see .toggle.on and .badge.purple there), which matches
  // design doc §2.7's "purple escalation badge" — so whatever the actual
  // rule ends up being, something like this keeps it consistent with the
  // rest of the app's palette rather than introducing a new one:
  //   .stage-escalation {
  //     display: flex; align-items: center; gap: 6px;
  //     font-size: 11.5px; font-weight: 500;
  //     color: var(--c-purple);
  //     background: var(--c-purple-dim);
  //     border: 1px solid var(--c-purple-border);
  //     border-radius: 6px; padding: 3px 8px; margin-bottom: 6px;
  //     width: fit-content;
  //   }
  // (.stage-swap's own rule — whatever file it's in — likely has its own
  // icon/spacing conventions worth matching too; this is a starting point,
  // not a replacement for checking that file once it's available.)
  const gpuMib = s.memory_mib && s.memory_mib.gpu_total;
  const memoryLabel = gpuMib
    ? (gpuMib >= 1024 ? `${(gpuMib / 1024).toFixed(2)} GiB` : `${Math.round(gpuMib)} MiB`)
    : null;
  const memoryBadge = memoryLabel
    ? `<span class="meta-item" title="GPU memory allocated for this model load">${memoryLabel} loaded</span>`
    : '';

  return `
    <div class="stage-node">
      <div class="stage-node-rail">
        <span class="stage-node-dot ${dotClass}"></span>
        ${isLast ? '' : '<span class="stage-node-line"></span>'}
      </div>
      <div class="stage-node-body">
        ${isEscalation
          ? `<div class="stage-escalation" title="This stage re-ran on a bigger model after truncation or low confidence on ${escapeHtml(prev.model)}">${icon('arrowUp') || '↑'} escalated from ${escapeHtml(prev.model)}</div>`
          : showSwap ? `<div class="stage-swap">${icon('swap')} swapped from ${escapeHtml(prev.model)}</div>` : ''}
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
          ${memoryBadge}
          ${thinkBar}
        </div>
      </div>
    </div>
  `;
}

// Track which artifact blocks are expanded, keyed by artifact name
// (e.g. "classification", "planspec"). Persists across re-renders within
// a mount so toggling one open while the SSE stream delivers new stages
// doesn't collapse it back.
let openArtifacts = new Set();
// Which iteration's snapshot is currently shown for a loop-writable
// artifact key (fixed/verdict/draft/critique — see write_iteration_artifact
// in clients/llm.py). Keyed by artifact key, value is an iteration number
// as a STRING (matches data.iterations' keys, which come from JSON object
// keys / directory names) or the sentinel 'latest' (data.artifacts[key] —
// today's un-versioned behaviour, and the only option for a single-pass
// run that never looped). Defaults to 'latest' for every key so a run
// with no correction loop renders exactly as before.
let selectedIteration = {};
// Loop-writable artifact keys that CAN have more than one iteration
// snapshot — the others (classification/planspec/appraisal_report/final/
// final_validation/run) are written at most once per run and have no
// iterations/ entry to select between. audit added alongside
// fixed/verdict/draft/critique once gatekeeper_node started writing
// audit.json per iteration too (nodes/gatekeeper.py) — previously the
// gatekeeper/audit stage produced real tokens/retries in the stages
// timeline but no artifact at all.
const ITERATION_AWARE_KEYS = new Set(['fixed', 'verdict', 'draft', 'critique', 'audit']);

function renderArtifacts() {
  const section = document.getElementById('artifacts-section');
  const mount_ = document.getElementById('artifacts-mount');
  if (!section || !mount_) return;

  const present = ARTIFACT_ORDER.filter((key) => data.artifacts[key] !== undefined);
  section.style.display = present.length ? '' : 'none';
  if (!present.length) return;

  // Available iteration numbers for a given key, oldest-first, as
  // strings — e.g. ['0','1','2','3'] for a run that looped 4 times.
  // Object.keys on data.iterations already comes back as strings (JSON
  // object keys), sorted numerically here since string sort would put
  // '10' before '2'.
  const iterationsFor = (key) => Object.keys(data.iterations)
    .filter((iter) => data.iterations[iter][key] !== undefined)
    .sort((a, b) => Number(a) - Number(b));

  mount_.innerHTML = present.map((key) => {
    const isOpen = openArtifacts.has(key);
    const iterOptions = ITERATION_AWARE_KEYS.has(key) ? iterationsFor(key) : [];
    // Only show a picker when there's genuinely something to pick between —
    // a single-pass run (iterOptions.length <= 1, since the one iteration
    // that exists is already identical to the flat "latest" file) renders
    // with no picker at all, unchanged from before this existed.
    const showPicker = iterOptions.length > 1;
    const selected = selectedIteration[key] || 'latest';
    const value = selected === 'latest'
      ? data.artifacts[key]
      : (data.iterations[selected] || {})[key];
    return `
      <div class="artifact-block" style="${present.indexOf(key) > 0 ? 'margin-top:10px;' : ''}">
        <button type="button" class="artifact-block-head" data-artifact-key="${escapeHtml(key)}" aria-expanded="${isOpen}">
          <span>${escapeHtml(titleCase(key))}<span class="mono" style="color:var(--text-disabled);font-weight:400;margin-left:8px;font-size:10.5px;">${escapeHtml(key)}.json</span></span>
          ${icon('chevronRight', 'chev')}
        </button>
        ${isOpen ? `
          ${showPicker ? `
            <div class="artifact-iteration-picker" style="display:flex;align-items:center;gap:6px;padding:8px 14px;border-top:1px solid var(--glass-border);flex-wrap:wrap;">
              <span style="font-size:10.5px;color:var(--text-tertiary);">Iteration</span>
              <div class="segmented" data-iteration-group="${escapeHtml(key)}">
                ${iterOptions.map((iter) => `<button type="button" data-iteration-value="${escapeHtml(iter)}" class="${selected === iter ? 'active' : ''}">${escapeHtml(iter)}</button>`).join('')}
                <button type="button" data-iteration-value="latest" class="${selected === 'latest' ? 'active' : ''}">Latest</button>
              </div>
            </div>
          ` : ''}
          <div class="artifact-block-body">${escapeHtml(prettyJson(value))}</div>
        ` : ''}
      </div>
    `;
  }).join('');

  mount_.querySelectorAll('[data-artifact-key]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.artifactKey;
      if (openArtifacts.has(key)) openArtifacts.delete(key);
      else openArtifacts.add(key);
      renderArtifacts();
    });
  });
  mount_.querySelectorAll('[data-iteration-group]').forEach((group) => {
    group.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-iteration-value]');
      if (!btn) return;
      const key = group.dataset.iterationGroup;
      selectedIteration[key] = btn.dataset.iterationValue;
      renderArtifacts();
    });
  });
}

// Manifest entries come from GET /run/{run_uuid} as artifacts.attachments —
// [{filename, size_bytes, path, excluded}, ...] (see server.py's
// _write_attachments / _update_attachment_manifest). "excluded" means the
// file is no longer folded into the pipeline's input on future turns, but
// the file itself and this manifest entry are kept — see
// DELETE /run/{run_uuid}/attachments/{filename} and its .../include
// counterpart in api.js.
function renderAttachments() {
  const mount_ = document.getElementById('attachments-mount');
  if (!mount_) return;

  const addLabel = document.getElementById('attachment-add-label');
  const addInput = document.getElementById('attachment-add-input');
  if (addLabel) {
    addLabel.innerHTML = attachmentUploadPending ? '… Adding' : `${icon('paperclip')} Add files`;
    addLabel.style.opacity = attachmentUploadPending ? '0.6' : '';
    addLabel.style.pointerEvents = attachmentUploadPending ? 'none' : '';
  }
  if (addInput) addInput.disabled = attachmentUploadPending;

  const manifest = data.artifacts.attachments || [];
  if (!manifest.length) {
    mount_.innerHTML = `<div class="text-tertiary" style="font-size:12px;">No files attached yet.</div>`;
    return;
  }

  mount_.innerHTML = manifest.map((a) => {
    const pending = attachmentActionPending === a.filename;
    const excluded = !!a.excluded;
    return `
      <div class="attachment-row" style="display:flex;align-items:center;gap:10px;padding:8px 0;${excluded ? 'opacity:0.55;' : ''}">
        ${icon('file')}
        <span class="mono" style="font-size:12px;flex:1;">${escapeHtml(a.filename)}</span>
        <span class="text-tertiary" style="font-size:11px;">${fmtNumber(a.size_bytes)} B</span>
        ${excluded ? `<span class="text-tertiary" style="font-size:11px;">Excluded</span>` : ''}
        <button
          type="button"
          class="btn-ghost"
          data-attachment-toggle="${escapeHtml(a.filename)}"
          data-attachment-excluded="${excluded}"
          ${pending ? 'disabled' : ''}
          style="padding:4px 10px;font-size:11px;"
          title="${excluded ? 'Re-include in future messages' : 'Remove from future messages'}"
        >
          ${pending ? '…' : (excluded ? 'Re-include' : 'Remove')}
        </button>
      </div>
    `;
  }).join('');

  mount_.querySelectorAll('[data-attachment-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => onAttachmentToggle(
      btn.dataset.attachmentToggle,
      btn.dataset.attachmentExcluded === 'true',
    ));
  });
}

async function onAttachmentToggle(filename, currentlyExcluded) {
  if (attachmentActionPending) return; // one in flight at a time
  attachmentActionPending = filename;
  renderAttachments();
  try {
    if (currentlyExcluded) {
      await includeAttachment(runUuid, filename);
    } else {
      await excludeAttachment(runUuid, filename);
    }
    await refreshArtifacts(); // pulls the updated manifest back down
    toastSuccess(currentlyExcluded ? `${filename} re-included.` : `${filename} removed from future messages.`);
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : `Couldn't update ${filename}.`);
  } finally {
    attachmentActionPending = null;
    renderAttachments();
  }
}

// Adds new file(s) to this run mid-chat via POST /run/{run_uuid}/attachments.
// Unlike sending a chat message, this is allowed regardless of run status —
// see the server-side comment on add_attachments — so no busy/terminal
// check is needed here the way onChatSubmit needs one.
async function onAttachmentAdd(e) {
  const files = Array.from(e.target.files || []);
  e.target.value = ''; // allow re-selecting the same file later
  if (!files.length || attachmentUploadPending) return;

  const toUpload = [];
  for (const file of files) {
    if (file.size > MAX_ATTACHMENT_BYTES) {
      toastError(`${file.name} is too large (max ${Math.round(MAX_ATTACHMENT_BYTES / 1024 / 1024)}MB).`);
      continue;
    }
    try {
      const content = await file.text();
      toUpload.push({ filename: file.name, content });
    } catch {
      toastError(`Couldn't read ${file.name}.`);
    }
  }
  if (!toUpload.length) return;

  attachmentUploadPending = true;
  renderAttachments();
  try {
    await addAttachments(runUuid, toUpload);
    await refreshArtifacts(); // pulls the updated manifest back down
    toastSuccess(toUpload.length > 1 ? `${toUpload.length} files added.` : `${toUpload[0].filename} added.`);
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : "Couldn't add file(s).");
  } finally {
    attachmentUploadPending = false;
    renderAttachments();
  }
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
    data.justAnsweredClarification = true;
    toastSuccess('Answer sent — run resumed.');
    renderStatusBadge();
    renderClarification();
    renderChat();
    openStream();
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Could not send answer.');
    btn.disabled = false;
    btn.textContent = 'Answer';
  } finally {
    clarifySubmitting = false;
  }
}

// Halted because a node's model call hit its output token cap (see
// server.py's POST /run/{run_uuid}/retry-truncated and
// pipeline/graph.py's _wrap_node_for_truncation_retry). Mirrors
// renderClarification/onClarifySubmit's structure — same mount-and-form
// pattern, same status/data shape — but shows what the model was
// actually doing when it ran out of room (data.truncation.thinking_block
// / partial_answer — see clients/llm.py's _extract_thinking_partial,
// which is what makes this content available at all instead of being
// silently discarded on truncation) and lets the person pick a new cap
// rather than free-text an answer.
// Tracks whether the thinking/partial-output preview is expanded.
// Simple boolean (not a Set like openArtifacts) since only one
// truncation widget is ever shown at a time.
let truncationPreviewOpen = false;

function renderTruncation() {
  const mount_ = document.getElementById('truncate-mount');
  if (!mount_) return;
  if (data.status !== 'waiting_for_truncation_retry' || !data.truncation) {
    mount_.innerHTML = '';
    return;
  }
  const t = data.truncation;
  const preview = (t.thinking_block || t.partial_answer || '').trim();
  const previewLabel = t.thinking_block ? 'What it was thinking' : 'Partial output';
  // Default the suggested retry cap to double whatever just failed —
  // matches the graph-side defensive fallback in
  // _wrap_node_for_truncation_retry (used only if no cap is supplied),
  // so the two stay in sync as a sensible default rather than picking
  // an unrelated number here.
  const suggestedCap = Math.max((Number(t.cap) || 0) * 2, (Number(t.cap) || 0) + 256, 512);
  mount_.innerHTML = `
    <div class="clarify-card">
      <div style="display:flex;align-items:center;gap:8px;">
        <span class="badge orange">Hit token limit</span>
        <span style="opacity:0.7;font-size:13px;">during "${escapeHtml(t.stage || 'unknown')}"</span>
      </div>
      <div class="clarify-question">
        Generated ${escapeHtml(String(t.tokens_out ?? '?'))} tokens and hit the cap
        (${escapeHtml(String(t.cap ?? '?'))}) before finishing.
      </div>
      ${preview ? `
        <div class="artifact-block">
          <button type="button" class="artifact-block-head" id="truncation-preview-toggle" aria-expanded="${truncationPreviewOpen}">
            <span>${escapeHtml(previewLabel)}<span class="mono" style="color:var(--text-disabled);font-weight:400;margin-left:8px;font-size:10.5px;">${preview.length.toLocaleString()} chars</span></span>
            ${icon('chevronRight', 'chev')}
          </button>
          ${truncationPreviewOpen ? `<div class="artifact-block-body">${escapeHtml(preview)}</div>` : ''}
        </div>
      ` : `
        <div style="opacity:0.7;font-size:13px;">No partial content was recovered.</div>
      `}
      <form class="clarify-form" id="truncate-form">
        <label for="truncate-cap" style="font-size:13px;opacity:0.8;">New token limit for this stage</label>
        <input
          type="number" class="text-input" id="truncate-cap"
          min="1" step="1" value="${suggestedCap}" required
        />
        <button type="submit" class="btn btn-primary" id="truncate-submit">Retry with higher limit</button>
      </form>
    </div>
  `;
  const previewToggle = document.getElementById('truncation-preview-toggle');
  if (previewToggle) {
    previewToggle.addEventListener('click', () => {
      truncationPreviewOpen = !truncationPreviewOpen;
      renderTruncation();
    });
  }
  document.getElementById('truncate-form').addEventListener('submit', onTruncationRetrySubmit);
}

async function onTruncationRetrySubmit(e) {
  e.preventDefault();
  if (truncationRetrySubmitting) return;
  const capInput = document.getElementById('truncate-cap');
  const outputCap = parseInt(capInput.value, 10);
  if (!outputCap || outputCap <= 0) {
    toastError('Enter a positive token limit.');
    return;
  }
  truncationRetrySubmitting = true;
  const btn = document.getElementById('truncate-submit');
  btn.disabled = true;
  btn.textContent = 'Retrying…';
  try {
    await retryTruncated(runUuid, outputCap);
    data.status = 'running';
    data.truncation = null;
    toastSuccess('Retrying with a higher token limit — run resumed.');
    renderStatusBadge();
    renderTruncation();
    renderChat();
    openStream();
  } catch (err) {
    toastError(err instanceof ApiError ? err.message : 'Could not retry.');
    btn.disabled = false;
    btn.textContent = 'Retry with higher limit';
  } finally {
    truncationRetrySubmitting = false;
  }
}

// ── Chat drawer ──────────────────────────────────────────────────────

function renderChat() {
  const handle = document.getElementById('chat-drawer-handle');
  const body = document.getElementById('chat-drawer-body');
  const drawer = document.getElementById('chat-drawer');
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send-btn');
  if (!handle || !body || !drawer) return;

  drawer.classList.toggle('expanded', drawerExpanded);
  const chevronEl = handle.querySelector('.chat-drawer-chevron');
  if (chevronEl) chevronEl.outerHTML = icon(drawerExpanded ? 'chevronDown' : 'chevronUp', 'chat-drawer-chevron');

  const msgs = chat.messages;
  const visible = drawerExpanded ? msgs : msgs.slice(-2);

  // Cheap fingerprint of what's about to be rendered. The 5s lesson poller
  // calls renderChat() far more often than the visible content actually
  // changes; rebuilding innerHTML unconditionally replayed each card's
  // fade-in animation and reset any in-progress scroll/selection, which
  // read as a periodic "flash". Only touch the DOM when something in the
  // fingerprint actually moved.
  const fingerprint = JSON.stringify([
    chat.loaded,
    drawerExpanded,
    openDetailSeq,
    visible.map((m) => [m.seq, m.content, (m.lessons_used || []).length]),
  ]);
  const contentChanged = fingerprint !== lastChatFingerprint;
  lastChatFingerprint = fingerprint;

  if (!contentChanged) {
    // Nothing to redraw, but gating below (input/send button) still needs
    // to reflect the latest status.
  } else if (!chat.loaded) {
    body.innerHTML = loadingRow('Loading conversation…');
  } else if (!visible.length) {
    body.innerHTML = emptyState({ iconName: 'runs', title: 'No messages yet' });
  } else {
    body.innerHTML = visible
      .map((m) => chatBubble(m, _triggeringUserSeq(msgs, msgs.indexOf(m))))
      .join('');
    body.querySelectorAll('[data-msg-seq]').forEach((el) => {
      el.addEventListener('click', () => {
        const seq = Number(el.dataset.msgSeq);
        openDetailSeq = openDetailSeq === seq ? null : seq;
        renderChat();
      });
    });
    body.querySelectorAll('[data-copy-seq]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation(); // don't also trigger the card's expand/collapse
        const seq = Number(btn.dataset.copySeq);
        const msg = msgs.find((m) => m.seq === seq);
        if (msg) copyToClipboard(msg.content, btn);
      });
    });
  }

  // Input gating, per the state machine in chat-ui-integration.md:
  // running / waiting_for_clarification -> disabled (clarification has
  // its own form above); complete/unresolvable/error/cancelled/interrupted
  // -> enabled, routed to POST /chat.
  const canSend = TERMINAL_STATUSES.includes(data.status) && !chatSending;
  if (input) {
    input.disabled = !canSend;
    input.placeholder = data.status === 'running' || data.status === 'pending'
      ? 'Waiting for the run to finish…'
      : data.status === 'waiting_for_clarification'
        ? 'Answer the question above first…'
        : 'Message…';
  }
  if (sendBtn) sendBtn.disabled = !canSend;
}

const NODE_ICON = {
  describe: 'info',
  clarify: 'zap',
  gatekeeper: 'checkCircle',
  plan: 'layers',
};

// Finds the seq of the user message that triggered the assistant message
// at `allMsgs[assistantIdx]`, scanning backward. This is the seq
// run_dir/turns/<seq>/ is actually keyed by server-side (see
// get_chat_turn_dir / _run_chat_turn_thread's turn_seq = user_seq in
// server.py) — NOT the assistant message's own seq, which is always one
// higher purely because chat_store.append_message auto-increments.
// Previously the artifact lookup used `data.chatArtifacts[String(m.seq)]`
// directly (the assistant's own seq), which never matched anything past
// the very first turn and silently fell back to data.artifacts (the
// original run's artifacts) for every later message — see the bug this
// fixes.
//
// Takes the FULL message list (msgs, not the possibly-windowed `visible`
// slice used when the drawer is collapsed — see msgs.slice(-2) above),
// so the triggering user message is still found even when it fell
// outside that window. Scans backward past intervening 'system' rows
// (e.g. the "(Retrying '<stage>' ...)" / "Resume failed: ..." messages
// the truncation-retry and clarify-resume flows can insert) rather than
// assuming a fixed seq-1 offset, since a system message landing between
// the user turn and its eventual assistant reply would otherwise throw
// a naive offset off by one.
function _triggeringUserSeq(allMsgs, assistantIdx) {
  for (let i = assistantIdx - 1; i >= 0; i--) {
    if (allMsgs[i].role === 'user') return allMsgs[i].seq;
  }
  return null;
}

function chatBubble(m, triggeringUserSeq) {
  if (m.role === 'system') {
    return `
      <div class="chat-system-notice">
        ${icon('alertTriangle')}
        <span>${escapeHtml(m.content)}</span>
      </div>
    `;
  }

  if (m.role === 'user') {
    return `
      <div class="chat-bubble-row user">
        <div class="chat-bubble user">
          <div class="chat-bubble-content">${escapeHtml(m.content)}</div>
          <div class="chat-bubble-time">${fmtRelativeTime(m.created_at)}</div>
        </div>
      </div>
    `;
  }

  // Assistant turn — nested card. Detail (artifacts if this turn produced
  // any, else the raw message metadata) expands inline inside the card,
  // per the Move History reference, rather than in a separate panel.
  // Lessons used on this turn also live inside that same expandable
  // detail panel, as their own labeled section — separate from the
  // artifact/metadata body, since a turn can have both.
  //
  // Artifact lookup: data.chatArtifacts[seq] holds this turn's OWN
  // artifact snapshot (run_dir/turns/<seq>/*.json — see get_chat_turn_dir
  // server-side), which is what makes classify/plan/draft/etc. show the
  // right content for message 5 instead of message 1's. The very first
  // assistant reply (produced by the original run, not a chat follow-up)
  // has no turns/ directory at all — its artifacts live at the top level
  // — so it's the one case that falls back to data.artifacts. Every
  // artifact key present for the turn is shown (ARTIFACT_ORDER order),
  // not just the single NODE_ARTIFACT_KEY-mapped one, since one chat
  // turn can produce classification+plan+draft+... in a single pass.
  //
  // IMPORTANT: the lookup key is triggeringUserSeq (the preceding user
  // message's seq — see _triggeringUserSeq above), NOT m.seq. The
  // per-turn directory server-side is keyed by the USER message's seq
  // (turn_seq = user_seq in _run_chat_turn_thread), while m here is the
  // ASSISTANT reply, whose seq is always one higher purely from
  // chat_store.append_message's auto-increment. Using m.seq directly
  // meant this never matched anything past the very first exchange and
  // silently fell back to data.artifacts (the original run) every time.
  const lessonsUsed   = m.lessons_used || [];
  const turnArtifacts = triggeringUserSeq != null ? data.chatArtifacts[String(triggeringUserSeq)] : undefined;
  const artifactSource = turnArtifacts || (turnArtifacts === undefined ? data.artifacts : null);
  const artifactKeys  = artifactSource
    ? ARTIFACT_ORDER.filter((key) => artifactSource[key] !== undefined)
    : [];
  const hasDetail = !!m.node_id || lessonsUsed.length > 0 || artifactKeys.length > 0;
  const isOpen = openDetailSeq === m.seq;
  // Legacy single-key fallback for the metadata label when nothing in
  // ARTIFACT_ORDER matched but NODE_ARTIFACT_KEY still names one (e.g. a
  // node whose artifact key isn't in ARTIFACT_ORDER for some reason).
  const legacyArtifactKey = NODE_ARTIFACT_KEY[m.node_id];
  const hasAnyArtifact = artifactKeys.length > 0 || (legacyArtifactKey && artifactSource && artifactSource[legacyArtifactKey] !== undefined);

  return `
    <div class="chat-bubble-row assistant">
      <div class="chat-card ${hasDetail ? 'clickable' : ''} ${isOpen ? 'open' : ''}" ${hasDetail ? `data-msg-seq="${m.seq}"` : ''}>
        ${m.node_id ? `
          <div class="chat-card-head">
            <span class="node-pill node-${escapeHtml(m.node_id)}">${icon(NODE_ICON[m.node_id] || 'info')}${escapeHtml(titleCase(m.node_id))}</span>
            <span class="chat-card-time">${fmtRelativeTime(m.created_at)}</span>
            <button type="button" class="btn-icon chat-copy-btn" data-copy-seq="${m.seq}" title="Copy message">⧉</button>
          </div>
          <div class="chat-card-content chat-markdown">${renderMarkdown(m.content)}</div>
        ` : `
          <div class="chat-card-content chat-markdown">
            <span class="chat-card-time-inline">${fmtRelativeTime(m.created_at)}</span>
            <button type="button" class="btn-icon chat-copy-btn" data-copy-seq="${m.seq}" title="Copy message">⧉</button>
            ${renderMarkdown(m.content)}
          </div>
        `}
        ${lessonsUsed.length ? `
          <div class="tag-row" style="margin-top:8px;">
            <span class="badge purple" title="Lessons injected into this turn's prompt">${icon('book')} ${lessonsUsed.length} lesson${lessonsUsed.length === 1 ? '' : 's'} used</span>
          </div>
        ` : ''}
        ${hasDetail ? `
          <div class="chat-card-expand-row">
            <span class="chev">${icon('chevronRight')}</span>
            <span>${hasAnyArtifact ? (artifactKeys.length > 1 ? `View ${artifactKeys.length} artifacts` : 'View details') : (m.node_id ? 'View metadata' : 'View lessons')}</span>
            ${m.run_iteration != null ? `<span style="margin-left:auto;">iteration ${m.run_iteration}</span>` : ''}
          </div>
        ` : ''}
        ${isOpen ? `
          ${artifactKeys.length ? artifactKeys.map((key) => `
            <div class="chat-card-detail">
              <div class="chat-card-detail-label">${escapeHtml(titleCase(key))}<span class="mono" style="color:var(--text-disabled);font-weight:400;margin-left:8px;font-size:10.5px;">${escapeHtml(key)}.json</span></div>
              <div class="chat-card-detail-body">${escapeHtml(prettyJson(artifactSource[key]))}</div>
            </div>
          `).join('') : (m.node_id ? `
            <div class="chat-card-detail">
              <div class="chat-card-detail-label">Message detail</div>
              <div class="chat-card-detail-body chat-markdown">${renderMarkdown(m.content)}</div>
            </div>
          ` : '')}
          ${lessonsUsed.length ? lessonsUsedDetail(lessonsUsed) : ''}
        ` : ''}
      </div>
    </div>
  `;
}

function lessonsUsedDetail(lessonsUsed) {
  return `
    <div class="chat-card-detail">
      <div class="chat-card-detail-label">Lessons used (${lessonsUsed.length})</div>
      <div style="display:flex;flex-direction:column;gap:10px;margin-top:6px;">
        ${lessonsUsed.map((l) => `
          <div class="lesson-mini-card">
            <div class="tag-row">
              <span class="tag">${escapeHtml(titleCase(l.issue_category || 'other'))}</span>
              <span class="tag">${escapeHtml(titleCase(l.task_type || ''))}</span>
              ${l.score != null ? `<span class="text-tertiary mono" style="font-size:11px;">score ${Number(l.score).toFixed(2)}</span>` : ''}
            </div>
            <div class="lesson-resolution" style="margin-top:6px;">${escapeHtml(l.resolution_pattern || '(lesson deleted since)')}</div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

async function onChatSubmit(e) {
  e.preventDefault();
  if (chatSending) return;
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message) return;

  chatSending = true;
  const sendBtn = document.getElementById('chat-send-btn');
  input.disabled = true;
  sendBtn.disabled = true;

  try {
    await sendChatMessage(
      runUuid, message, chatReplan, chatReplan ? (chatProfile || null) : null,
      null, chatUseSearch, chatHumanInTheLoop,
    );
    input.value = '';
    data.status = 'running';
    renderStatusBadge();
    renderTimeline();
    openStream();
    await loadChat(); // pick up the new user turn immediately
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      toastError("This run is still busy — wait for it to finish before sending another message.");
    } else {
      toastError(err instanceof ApiError ? err.message : 'Could not send message.');
    }
  } finally {
    chatSending = false;
    renderChat();
  }
}