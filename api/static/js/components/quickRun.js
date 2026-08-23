// Compact top-bar "start a run" trigger, styled after Dacelo's
// ConnectionToolbarItem: a small dot that pulses while busy, a label that
// goes bold while unsettled and relaxes once settled, plain button style.
// It doesn't submit a task itself (task text lives on the Runs view) —
// tapping it jumps to #/runs and focuses the task field, unless a run is
// already in flight from a prior tap, mirroring the SwiftUI guard.

import { icon } from '../icons.js';

let mountEl = null;
let launching = false; // mirrors `isConnecting` in the reference

export function initQuickRun(el) {
  mountEl = el;
  render();
}

// Called by runs.js right before it calls POST /run, and again once that
// call settles — this is what actually drives the pulse, same as the
// SwiftUI version's Task { ...; sleep; isConnecting = false }.
export function setLaunching(on) {
  launching = on;
  render();
}

function render() {
  if (!mountEl) return;
  const dotColor = launching ? 'blue' : 'grey';
  const label = launching ? 'Starting…' : 'New run';

  mountEl.innerHTML = `
    <button class="quick-run-btn" id="quick-run-btn" ${launching ? 'disabled' : ''} title="Start a new run">
      <span class="status-dot ${dotColor}${launching ? ' pulsing' : ''}"></span>
      <span class="quick-run-label ${launching ? 'unsettled' : 'settled'}">${label}</span>
      ${icon('play', 'quick-run-icon')}
    </button>
  `;

  document.getElementById('quick-run-btn').addEventListener('click', () => {
    if (launching) return; // guard !isConnecting else return
    if (location.hash.replace(/^#\/?/, '') === 'runs' || location.hash === '') {
      document.getElementById('task-input')?.focus();
    } else {
      location.hash = '#/runs';
      // Focus after the runs view has had a chance to mount.
      setTimeout(() => document.getElementById('task-input')?.focus(), 60);
    }
  });
}
