import { icon } from './icons.js';
import { escapeHtml } from './format.js';

let root = null;

function ensureRoot() {
  if (!root) root = document.getElementById('toast-root');
  return root;
}

export function toast(message, type = 'info', timeout = 4200) {
  const r = ensureRoot();
  if (!r) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const iconName = type === 'error' ? 'xCircle' : type === 'success' ? 'checkCircle' : 'info';
  el.innerHTML = `${icon(iconName, 'toast-icon')}<span>${escapeHtml(message)}</span>`;
  r.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 0.25s ease';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 250);
  }, timeout);
}

export const toastError = (msg) => toast(msg, 'error');
export const toastSuccess = (msg) => toast(msg, 'success');
