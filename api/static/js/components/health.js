import { getHealth, ApiError } from '../api.js';
import { escapeHtml } from '../format.js';

const POLL_MS = 4000;

let mountEl = null;
let timer = null;
let lastHealth = null;
let hasConnectedOnce = false;
let listeners = [];

export function onHealthChange(fn) {
  listeners.push(fn);
  return () => { listeners = listeners.filter((f) => f !== fn); };
}

export function getLastHealth() {
  return lastHealth;
}

export function initHealth(el) {
  mountEl = el;
  render();
  poll();
  timer = setInterval(poll, POLL_MS);
}

async function poll() {
  // Only show the "connecting" state on the very first load, or after we've
  // dropped a connection — not on every routine successful re-poll, which
  // was causing the pill to flash back to "Connecting…" every 4s.
  if (!hasConnectedOnce) render();
  try {
    const h = await getHealth();
    lastHealth = { ...h, ok: true };
    hasConnectedOnce = true;
  } catch (err) {
    lastHealth = { ok: false, error: err instanceof ApiError ? err.message : String(err) };
    hasConnectedOnce = false;
  }
  render();
  listeners.forEach((fn) => fn(lastHealth));
}

function render() {
  if (!mountEl) return;

  let dotClass = 'yellow breathing-in';
  let label = 'Connecting…';
  let modelHtml = '';
  let queueHtml = '';

  if (lastHealth) {
    if (lastHealth.ok) {
      const model = lastHealth.hot_model
        ? escapeHtml(lastHealth.hot_model)
        : 'no model loaded';
      modelHtml = `<span class="health-model mono">${model}</span>`;
      if (lastHealth.queue_depth > 0) {
        queueHtml = `<span class="health-queue">queued · ${lastHealth.queue_depth} ahead</span>`;
      }
      if (lastHealth.active_runs > 0) {
        dotClass = 'blue breathing-in';
        label = 'Running';
      } else {
        dotClass = 'green';
        label = 'Connected';
      }
    } else {
      dotClass = 'red';
      label = 'Disconnected';
    }
  }

  const labelClass = (!lastHealth || !lastHealth.ok) ? 'unsettled' : 'settled';

  mountEl.innerHTML = `
    <div class="health-pill" title="${lastHealth && !lastHealth.ok ? escapeHtml(lastHealth.error || '') : ''}">
      <span class="status-dot ${dotClass}"></span>
      <span class="health-label ${labelClass}">${label}</span>
      ${modelHtml}
      ${queueHtml}
    </div>
  `;
}