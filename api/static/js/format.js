export function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function fmtNumber(n) {
  if (n == null) return '—';
  return n.toLocaleString('en-US');
}

export function fmtMs(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function fmtDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function fmtRelativeTime(isoString) {
  if (!isoString) return '—';
  const then = new Date(isoString.endsWith('Z') || isoString.includes('+') ? isoString : isoString + 'Z');
  const diffMs = Date.now() - then.getTime();
  const diffSec = Math.round(diffMs / 1000);
  if (Number.isNaN(diffSec)) return isoString;
  if (diffSec < 5) return 'just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.round(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.round(diffHr / 24);
  return `${diffDay}d ago`;
}

// Derive a run's duration in seconds from whatever timestamps are available.
// The /runs list normally supplies total_latency_ms directly (preferred —
// see runs.js), but that field is nullable per the contract, so this is the
// wall-clock fallback: started_at vs completed_at. completed_at is itself
// sometimes null on a finished run (a fixture gap in mock_server.py), in
// which case there's nothing reliable to compute from and we show "—".
export function runDurationSeconds(run) {
  if (!run.started_at || !run.completed_at) return null;
  const start = parseTs(run.started_at);
  const end = parseTs(run.completed_at);
  return (end - start) / 1000;
}

function parseTs(s) {
  const iso = s.endsWith('Z') || s.includes('+') ? s : s + 'Z';
  return new Date(iso).getTime();
}

export function prettyJson(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

export function titleCase(s) {
  if (!s) return '';
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

// Status → semantic color name, shared across run list / badges / dots.
export function statusColor(status) {
  switch (status) {
    case 'complete': return 'green';
    case 'running': return 'blue';
    case 'pending': return 'blue';
    case 'waiting_for_clarification': return 'purple';
    case 'unresolvable': return 'orange';
    case 'error': return 'red';
    case 'cancelled': return 'grey';
    case 'interrupted': return 'grey';
    default: return 'grey';
  }
}
