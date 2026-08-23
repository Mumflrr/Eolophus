let el = null;

export function setUltraAmbient(on) {
  if (!el) el = document.getElementById('gradient-wash');
  if (!el) return;
  el.classList.toggle('ultra', !!on);
}
