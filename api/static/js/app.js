import { icon } from './icons.js';
import { initHealth } from './components/health.js';
//import { initQuickRun } from './components/quickRun.js';
import * as runsView from './views/runs.js';
import * as runDetailView from './views/runDetail.js';
import * as modelsView from './views/models.js';
import * as pipelinesView from './views/pipelines.js';
import * as pipelineEditorView from './views/pipelineEditor.js';
import * as lessonsView from './views/lessons.js';
import * as settingsView from './views/settings.js';

const NAV = [
  { id: 'runs', label: 'Runs', iconName: 'runs', match: (h) => h === '' || h.startsWith('runs') },
  { id: 'models', label: 'Models', iconName: 'cpu', match: (h) => h.startsWith('models') },
  { id: 'pipelines', label: 'Pipelines', iconName: 'git', match: (h) => h.startsWith('pipelines') },
  { id: 'lessons', label: 'Lessons', iconName: 'book', match: (h) => h.startsWith('lessons') },
  { id: 'settings', label: 'Settings', iconName: 'gear', match: (h) => h.startsWith('settings') },
];

let currentView = null;

function renderShell() {
  document.getElementById('app').innerHTML = `
    <nav class="sidebar" aria-label="Primary">
      <div class="brand">
        <div class="brand-mark">E</div>
        <div class="brand-text">
          <div class="brand-title">Eolophus</div>
          <div class="brand-sub">Pipeline Control</div>
        </div>
      </div>
      <div class="nav-group" id="nav-group-desktop"></div>
    </nav>
    <div class="main-col">
      <header class="topbar">
        <div class="topbar-title" id="topbar-title">Runs</div>
        <div class="topbar-actions">
          <div id="health-mount-top"></div>
        </div>
      </header>
      <main id="view-root"></main>
    </div>
    <nav class="bottom-nav" id="nav-group-mobile" aria-label="Primary"></nav>
  `;
}

function renderNav(activeId) {
  const itemHtml = (item) => `
    <button class="nav-item ${item.id === activeId ? 'active' : ''}" data-nav="${item.id}">
      <span class="nav-icon">${icon(item.iconName)}</span>
      <span>${item.label}</span>
    </button>
  `;
  document.getElementById('nav-group-desktop').innerHTML = NAV.map(itemHtml).join('');
  document.getElementById('nav-group-mobile').innerHTML = NAV.map(itemHtml).join('');
  document.querySelectorAll('[data-nav]').forEach((btn) => {
    btn.addEventListener('click', () => {
      location.hash = `#/${btn.dataset.nav}`;
    });
  });
}

const TITLES = {
  runs: 'Runs',
  models: 'Model Management',
  pipelines: 'Pipelines',
  lessons: 'Lessons',
  settings: 'Settings',
};

function route() {
  const hash = location.hash.replace(/^#\/?/, '');
  const segments = hash.split('/').filter(Boolean);
  const top = segments[0] || 'runs';

  if (currentView && typeof currentView.unmount === 'function') {
    currentView.unmount();
    currentView = null;
  }

  const active = NAV.find((n) => n.match(hash)) || NAV[0];
  renderNav(active.id);

  const root = document.getElementById('view-root');
  root.scrollTop = 0;

  if (top === 'runs' && segments[1]) {
    currentView = runDetailView;
    document.getElementById('topbar-title').textContent = 'Run detail';
    runDetailView.mount(root, { runUuid: segments[1] });
  } else if (top === 'runs') {
    currentView = runsView;
    document.getElementById('topbar-title').textContent = TITLES.runs;
    runsView.mount(root);
  } else if (top === 'models') {
    currentView = modelsView;
    document.getElementById('topbar-title').textContent = TITLES.models;
    modelsView.mount(root);
  } else if (top === 'pipelines' && segments[1]) {
    currentView = pipelineEditorView;
    document.getElementById('topbar-title').textContent = segments[1] === 'new' ? 'New pipeline' : segments[1];
    pipelineEditorView.mount(root, { name: segments[1] === 'new' ? null : decodeURIComponent(segments[1]) });
  } else if (top === 'pipelines') {
    currentView = pipelinesView;
    document.getElementById('topbar-title').textContent = TITLES.pipelines;
    pipelinesView.mount(root);
  } else if (top === 'lessons') {
    currentView = lessonsView;
    document.getElementById('topbar-title').textContent = TITLES.lessons;
    lessonsView.mount(root);
  } else if (top === 'settings') {
    currentView = settingsView;
    document.getElementById('topbar-title').textContent = TITLES.settings;
    settingsView.mount(root);
  } else {
    currentView = runsView;
    document.getElementById('topbar-title').textContent = TITLES.runs;
    runsView.mount(root);
  }
}

function init() {
  renderShell();
  initHealth(document.getElementById('health-mount-top'));
//  initQuickRun(document.getElementById('quick-run-mount'));

  window.addEventListener('hashchange', route);
  route();
}

document.addEventListener('DOMContentLoaded', init);
