// Small dependency-free outline icon set — Lucide-style, 24x24 source
// geometry, rendered thin and small by default so it reads as an accent
// next to text rather than a dominant shape. Kept local so the app has
// zero external runtime dependencies.

const PATHS = {
  play: '<path d="M6 4l14 8-14 8V4z"/>',
  runs: '<path d="M4 6h16M4 12h16M4 18h10"/>',
  cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  sliders: '<path d="M4 6h6M14 6h6M4 12h10M18 12h2M4 18h2M10 18h10"/><circle cx="12" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="8" cy="18" r="2"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09a1.65 1.65 0 00-1.08-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09a1.65 1.65 0 001.51-1.08 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/>',
  chevronRight: '<path d="M9 6l6 6-6 6"/>',
  chevronDown: '<path d="M6 9l6 6 6-6"/>',
  alertTriangle: '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><path d="M12 9v4M12 17h.01"/>',
  check: '<path d="M20 6L9 17l-5-5"/>',
  checkCircle: '<circle cx="12" cy="12" r="10"/><path d="M8 12l3 3 5-6"/>',
  x: '<path d="M18 6L6 18M6 6l12 12"/>',
  xCircle: '<circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/>',
  refresh: '<path d="M21 12a9 9 0 11-3.02-6.71M21 4v5h-5"/>',
  zap: '<path d="M13 2L4 14h6l-1 8 9-12h-6l1-8z"/>',
  trash: '<path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
  wifi: '<path d="M2 8.5a16 16 0 0120 0M5 12a11 11 0 0114 0M8.5 15.5a6 6 0 017 0"/><circle cx="12" cy="19" r="1"/>',
  swap: '<path d="M17 3l4 4-4 4M3 7h18M7 21l-4-4 4-4M21 17H3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>',
  layers: '<path d="M12 2l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5M3 17l9 5 9-5"/>',
  arrowLeft: '<path d="M19 12H5M12 19l-7-7 7-7"/>',
  externalLink: '<path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><path d="M15 3h6v6M10 14L21 3"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  minus: '<path d="M5 12h14"/>',
  reload: '<path d="M4 4v5h5M20 20v-5h-5"/><path d="M4 9a9 9 0 0114.13-5.66M20 15a9 9 0 01-14.13 5.66"/>',
  power: '<path d="M12 2v9"/><path d="M18.4 6.6a9 9 0 11-12.8 0"/>',
  edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4 12.5-12.5z"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>',
  arrowRight: '<path d="M5 12h14M12 5l7 7-7 7"/>',
  git: '<circle cx="6" cy="6" r="2.2"/><circle cx="6" cy="18" r="2.2"/><circle cx="18" cy="12" r="2.2"/><path d="M6 8.2V15.8M8 6.5c4.5 0 4.5 5.5 8 5.5"/>',
};

// Base geometry every icon is authored against — kept out of the raw
// PATHS strings so sizing lives in exactly one place.
const VIEWBOX = '0 0 24 24';
const DEFAULT_STROKE = 1.6;

export function icon(name, cls = '', opts = {}) {
  const p = PATHS[name] || PATHS.info;
  const stroke = opts.stroke || DEFAULT_STROKE;
  const classes = ['icon', cls].filter(Boolean).join(' ');
  return `<svg class="${classes}" viewBox="${VIEWBOX}" fill="none" stroke="currentColor" stroke-width="${stroke}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${p}</svg>`;
}
