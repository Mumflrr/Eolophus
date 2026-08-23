import { icon } from '../icons.js';
import { escapeHtml } from '../format.js';

export function cardHeader({ icon: iconName, color = 'purple', title, subtitle = '', actionHtml = '' }) {
  return `
    <div class="card-header">
      ${iconName ? `<div class="card-icon-badge ${color}">${icon(iconName)}</div>` : ''}
      <div class="card-header-text">
        <div class="card-title">${escapeHtml(title)}</div>
        ${subtitle ? `<div class="card-subtitle">${escapeHtml(subtitle)}</div>` : ''}
      </div>
      ${actionHtml ? `<div class="card-header-action">${actionHtml}</div>` : ''}
    </div>
  `;
}

export function cardRow({ title, sub = '', controlHtml = '' }) {
  return `
    <div class="card-row">
      <div class="card-row-label">
        <span class="row-title">${escapeHtml(title)}</span>
        ${sub ? `<span class="row-sub">${escapeHtml(sub)}</span>` : ''}
      </div>
      <div class="card-row-control">${controlHtml}</div>
    </div>
  `;
}

export function emptyState({ iconName = 'info', title, sub = '' }) {
  return `
    <div class="empty-state">
      ${icon(iconName)}
      <div class="empty-state-title">${escapeHtml(title)}</div>
      ${sub ? `<div class="empty-state-sub">${escapeHtml(sub)}</div>` : ''}
    </div>
  `;
}

export function loadingRow(text = 'Loading…') {
  return `<div class="loading-row"><span class="spinner"></span>${escapeHtml(text)}</div>`;
}
