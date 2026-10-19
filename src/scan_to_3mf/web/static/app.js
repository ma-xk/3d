import { DualViewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const HEADERS = { 'X-Scan-To-3MF': '1' };
const EXTENSIONS = ['.stl', '.obj', '.ply', '.3mf'];
const UNIT_ORDER = ['mm', 'cm', 'm', 'in', 'ft', 'um'];
const ROWS = [
  ['Triangles', 'triangles', 'int'],
  ['Shells', 'shells', 'shells'],
  ['Holes', 'holes', 'zero'],
  ['Boundary (open) edges', 'boundary_edges', 'zero'],
  ['Non-manifold edges', 'non_manifold_edges', 'zero'],
  ['Non-manifold vertices', 'non_manifold_vertices', 'zero'],
  ['Degenerate triangles', 'degenerate_faces', 'zero'],
  ['Duplicate triangles', 'duplicate_faces', 'zero'],
  ['Self-intersecting triangles', 'self_intersecting_faces', 'zero'],
  ['Normals', 'orientation', 'orientation'],
  ['Watertight', 'watertight', 'bool'],
  ['Volume', 'volume_mm3', 'volume'],
  ['Bounding box (mm)', 'extents', 'bbox'],
];

const state = {
  info: null,
  job: null,
  poll: null,
  shownBefore: null,
  shownAfter: null,
  resultKey: null,
};
const viewer = new DualViewer($('pane-before'), $('pane-after'));

// ---------------------------------------------------------------- helpers

const fmtInt = (n) => (n === null || n === undefined ? '—' : Number(n).toLocaleString('en-US'));
const fmtLen = (x) => (Math.abs(x) >= 100 ? x.toFixed(1) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toPrecision(3));
const fmtBox = (e) => e.map(fmtLen).join(' × ');

function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 6000);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return res;
}

function unitFactor(key) { return state.info?.units?.[key]?.factor ?? 1; }
function unitLabel(key) { return state.info?.units?.[key]?.label ?? key; }

function effectiveUnit() {
  const insp = state.job?.inspect;
  if (insp?.file_unit && !$('override-unit').checked) return insp.file_unit;
  return $('unit').value || 'mm';
}

// ---------------------------------------------------------------- init

async function init() {
  try {
    state.info = await (await api('/api/info')).json();
  } catch (err) {
    toast(`Cannot reach the local server: ${err.message}`);
    return;
  }
  $('version').textContent = `v${state.info.version}`;
  const sel = $('unit');
  for (const key of UNIT_ORDER) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = `${unitLabel(key)} (${key === 'um' ? 'µm' : key})`;
    sel.appendChild(opt);
  }
  sel.value = 'mm';
  const d = state.info.defaults;
  $('opt-hole-mm').value = d.max_hole_mm;
  $('opt-shell-pct').value = +(d.min_shell_fraction * 100).toFixed(2);
  updateSlicer();
}

async function updateSlicer() {
  if (!state.info.slicer_checked) {
    setTimeout(async () => {
      try { state.info = await (await api('/api/info')).json(); } catch { /* retry */ }
      updateSlicer();
    }, 800);
    return;
  }
  const chip = $('slicer-chip');
  if (state.info.slicer && state.info.slicer.can_slice) {
    chip.textContent = `Slice test: ${state.info.slicer.label}`;
    chip.className = 'chip ok';
    $('slicer-name').textContent = state.info.slicer.label;
  } else if (state.info.slicer) {
    chip.textContent = `${state.info.slicer.label}: size check only`;
    chip.className = 'chip ok';
    $('slicer-name').textContent = `${state.info.slicer.label} (size check; install PrusaSlicer to slice)`;
  } else {
    chip.textContent = 'No slicer found: slice test skipped';
    chip.className = 'chip missing';
    $('slicer-name').textContent = 'PrusaSlicer (not found)';
    $('opt-slice').checked = false;
    $('opt-slice').disabled = true;
  }
}

// ---------------------------------------------------------------- upload

function pickFile(file) {
  if (!file) return;
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if (!EXTENSIONS.includes(ext)) {
    toast(`${file.name}: unsupported type. Use STL, OBJ, PLY, or 3MF.`);
    return;
  }
  upload(file);
}

async function upload(file) {
  stopPolling();
  resetForNewFile(file);
  try {
    const res = await api(`/api/upload?name=${encodeURIComponent(file.name)}`, {
      method: 'POST', headers: { ...HEADERS, 'Content-Type': 'application/octet-stream' }, body: file,
    });
    state.job = await res.json();
    startPolling();
  } catch (err) {
    showProgress(false);
    toast(`Upload failed: ${err.message}`);
  }
}

function resetForNewFile(file) {
  state.job = null;
  state.shownBefore = null;
  state.shownAfter = null;
  state.resultKey = null;
  viewer.clear('before');
  viewer.clear('after');
  viewer.fitted = false;
  $('empty-before').hidden = false;
  $('empty-before').textContent = 'Reading…';
  $('empty-after').hidden = false;
  $('dims-before').textContent = '';
  $('dims-after').textContent = '';
  $('file-info').hidden = false;
  $('file-name').textContent = file.name;
  $('file-meta').textContent = `${(file.size / 1e6).toFixed(1)} MB — reading and measuring…`;
  $('file-notes').innerHTML = '';
  $('dropzone').classList.add('compact');
  $('dropzone').querySelector('.drop-title').textContent = 'Drop another scan to replace it';
  $('result-card').hidden = true;
  $('report-card').hidden = true;
  $('clean').disabled = true;
  $('unit').disabled = true;
  $('unit-hint').hidden = true;
  $('size-mm').textContent = '—';
  $('size-in').textContent = '';
  showProgress(true, 'Reading and measuring the mesh…', null);
}

// ---------------------------------------------------------------- polling

function startPolling() {
  stopPolling();
  const tick = async () => {
    if (!state.job) return;
    try {
      const res = await api(`/api/jobs/${state.job.id}`);
      state.job = await res.json();
      render();
      if (['inspecting', 'cleaning'].includes(state.job.state)) state.poll = setTimeout(tick, 350);
    } catch (err) {
      toast(err.message);
    }
  };
  tick();
}

function stopPolling() {
  clearTimeout(state.poll);
  state.poll = null;
}

function render() {
  const job = state.job;
  if (!job) return;
  if (job.state === 'error') {
    showProgress(false);
    $('empty-before').textContent = 'Could not read this file';
    $('file-meta').textContent = job.error;
    $('clean').disabled = !job.inspect;
    toast(job.error);
    return;
  }
  if (job.inspect && state.shownBefore !== job.id) {
    state.shownBefore = job.id;
    renderInspect(job.inspect);
    loadPreview('before', job.inspect);
  }
  if (job.state === 'inspecting') {
    showProgress(true, 'Reading and measuring the mesh…', null);
  } else if (job.state === 'cleaning') {
    const p = job.progress || { message: 'Starting', fraction: 0 };
    showProgress(true, p.message + '…', p.fraction);
    $('clean').disabled = true;
  } else if (job.state === 'ready') {
    showProgress(false);
    $('clean').disabled = false;
  } else if (job.state === 'done' && job.result) {
    showProgress(false);
    $('clean').disabled = false;
    if (state.shownAfter !== job.result) {
      state.shownAfter = job.result;
      renderResult(job.result);
    }
  }
}

function showProgress(show, text = '', fraction = null) {
  $('progress').hidden = !show;
  if (!show) return;
  $('progress-text').textContent = text;
  const fill = $('progress-fill');
  fill.classList.toggle('indeterminate', fraction === null);
  if (fraction !== null) fill.style.width = `${Math.round(fraction * 100)}%`;
}

async function loadPreview(which, inspect) {
  try {
    const buf = await (await api(`/api/jobs/${state.job.id}/preview/${which}`)).arrayBuffer();
    let scale = 1;
    if (which === 'before' && inspect) scale = unitFactor(effectiveUnit()) / unitFactor(inspect.unit);
    viewer.setMesh(which, buf, { scale, refit: which === 'after' });
    $(`empty-${which}`).hidden = true;
  } catch (err) {
    toast(`Preview failed: ${err.message}`);
  }
}

// ---------------------------------------------------------------- inspect view

function renderInspect(insp) {
  $('file-meta').textContent = `${insp.format.toUpperCase()} · ${fmtInt(insp.file_triangles)} triangles`;
  $('file-notes').innerHTML = '';
  for (const note of insp.notes) {
    const li = document.createElement('li');
    li.textContent = note;
    $('file-notes').appendChild(li);
  }
  const sel = $('unit');
  if (insp.file_unit) {
    $('unit-file-note').hidden = false;
    $('unit-file-note').textContent = `This 3MF declares ${unitLabel(insp.file_unit)}; that unit is used.`;
    $('override-wrap').hidden = false;
    $('override-unit').checked = false;
    sel.value = insp.file_unit;
    sel.disabled = true;
    $('unit-label').textContent = 'Unit declared by the file';
  } else {
    $('unit-file-note').hidden = true;
    $('override-wrap').hidden = true;
    sel.disabled = false;
    $('unit-label').textContent = `${insp.format.toUpperCase()} has no units. The numbers are in`;
  }
  updateSize();
  $('report-card').hidden = false;
  renderMetrics(scaledMetrics(insp.metrics, lengthScale()), null);
  renderList('changes', [], 'Run a clean to see what changes.');
  renderRemaining(null);
  renderChecks([]);
  renderHoles(insp.holes, null, lengthScale());
  renderShells([]);
  $('not-checked').innerHTML = '';
  $('timings').textContent = '';
}

function lengthScale() {
  const insp = state.job?.inspect;
  if (!insp) return 1;
  return unitFactor(effectiveUnit()) / unitFactor(insp.unit);
}

function scaledMetrics(m, s) {
  if (s === 1) return m;
  return {
    ...m,
    extents: m.extents.map((x) => x * s),
    volume_mm3: m.volume_mm3 === null ? null : m.volume_mm3 * s * s * s,
  };
}

function updateSize() {
  const insp = state.job?.inspect;
  if (!insp) return;
  const unit = effectiveUnit();
  const mm = insp.raw_extents.map((x) => x * unitFactor(unit));
  $('size-mm').textContent = `${fmtBox(mm)} mm`;
  $('size-in').textContent = `${mm.map((x) => fmtLen(x / 25.4)).join(' × ')} in`;
  $('dims-before').textContent = `${fmtBox(mm)} mm`;
  renderUnitHint(insp.raw_extents, unit, Boolean(insp.file_unit && !$('override-unit').checked));
}

function renderUnitHint(raw, unit, fromFile) {
  const size = Math.max(...raw) * unitFactor(unit);
  const hint = $('unit-hint');
  hint.innerHTML = '';
  if (size >= 3 && size <= 2000) { hint.hidden = true; return; }
  const what = size < 3 ? 'unusually small' : 'unusually large';
  const p = document.createElement('div');
  p.textContent = `${fmtLen(size)} mm across is ${what}. ${fromFile ? 'The 3MF may declare the wrong unit.' : 'Is the unit right?'}`;
  hint.appendChild(p);
  for (const key of ['mm', 'cm', 'm', 'in']) {
    if (key === unit) continue;
    const s = Math.max(...raw) * unitFactor(key);
    if (s < 10 || s > 600) continue;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'btn secondary small';
    b.textContent = `Use ${unitLabel(key)} → ${Math.round(s)} mm`;
    b.addEventListener('click', () => {
      if (fromFile) $('override-unit').checked = true;
      $('unit').disabled = false;
      $('unit').value = key;
      onUnitChange();
    });
    hint.appendChild(b);
  }
  hint.hidden = false;
}

function onUnitChange() {
  updateSize();
  const insp = state.job?.inspect;
  if (!insp) return;
  if (!state.job.result || state.job.state !== 'done') {
    renderMetrics(scaledMetrics(insp.metrics, lengthScale()), null);
    renderHoles(insp.holes, null, lengthScale());
    viewer.setScale('before', lengthScale());
  }
  markStale();
}

function markStale() {
  if (!state.job?.result) return;
  $('stale').hidden = JSON.stringify(collectOptions()) === state.resultKey;
}

// ---------------------------------------------------------------- clean

function collectOptions() {
  const insp = state.job?.inspect;
  const override = Boolean(insp?.file_unit && $('override-unit').checked);
  return {
    unit: insp?.file_unit && !override ? null : $('unit').value,
    override_file_unit: override,
    remove_small_shells: $('opt-shells').checked,
    min_shell_fraction: Math.max(0, Number($('opt-shell-pct').value) || 0) / 100,
    fill_holes: $('opt-holes').checked,
    max_hole_mm: Math.max(0, Number($('opt-hole-mm').value) || 0),
    fill_large_holes: $('opt-large').checked,
    decimate_to: $('opt-decimate').checked ? Math.round(Number($('opt-target').value)) : null,
    place_on_bed: $('opt-bed').checked,
    slicer_check: $('opt-slice').checked,
  };
}

async function clean() {
  if (!state.job) return;
  const options = collectOptions();
  $('clean').disabled = true;
  $('result-card').hidden = true;
  showProgress(true, 'Starting…', 0.01);
  try {
    const res = await api(`/api/jobs/${state.job.id}/clean`, {
      method: 'POST', headers: { ...HEADERS, 'Content-Type': 'application/json' }, body: JSON.stringify(options),
    });
    state.job = await res.json();
    state.resultKey = JSON.stringify(options);
    state.shownAfter = null;
    startPolling();
  } catch (err) {
    showProgress(false);
    $('clean').disabled = false;
    toast(err.message);
  }
}

// ---------------------------------------------------------------- results

function renderResult(r) {
  $('result-card').hidden = false;
  $('stale').hidden = true;
  const verdict = $('verdict');
  verdict.textContent = r.verdict;
  const sliced = r.slicer && r.slicer.ok;
  verdict.className = `verdict ${r.printable ? (sliced || !r.slicer || r.slicer.ok === null ? 'good' : 'warn') : 'bad'}`;
  const dl = $('download');
  dl.href = `/api/jobs/${state.job.id}/download`;
  dl.setAttribute('download', r.output ? r.output.path.split(/[\\/]/).pop() : 'scan-clean.3mf');
  $('download-report').href = `/api/jobs/${state.job.id}/report`;
  $('download-report').setAttribute('download', '');
  if (r.output) $('output-meta').textContent = `${(r.output.bytes / 1e6).toFixed(1)} MB · ${fmtInt(r.output.triangles)} triangles · millimeters`;

  $('report-card').hidden = false;
  renderMetrics(r.before, r.after);
  renderSteps(r.steps);
  renderRemaining(r.remaining);
  renderChecks(r.verification, r.slicer);
  renderHoles(r.before_holes, r.after_holes, 1);
  renderShells(r.removed_shells);
  $('not-checked').innerHTML = r.not_checked.map((t) => `<li>${escapeHtml(t)}</li>`).join('');
  const t = r.timings;
  const mem = t.peak_memory_mb ? ` · peak memory ${Math.round(t.peak_memory_mb)} MB` : '';
  const slice = t.slice ? ` (slice test ${t.slice.toFixed(1)} s)` : '';
  $('timings').textContent = `Processed in ${t.total.toFixed(1)} s${slice}${mem} · scan-to-3mf ${r.version}`;

  $('dims-before').textContent = `${fmtBox(r.before.extents)} mm`;
  $('dims-after').textContent = `${fmtBox(r.after.extents)} mm`;
  loadPreview('before', null).then(() => loadPreview('after', null));
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function cell(m, key, kind) {
  if (!m) return { text: '', cls: 'pending' };
  const v = m[key];
  if (v === null || v === undefined) return { text: '—', cls: 'pending' };
  switch (kind) {
    case 'int': return { text: fmtInt(v), cls: '' };
    case 'zero': return { text: fmtInt(v), cls: v === 0 ? 'good' : 'bad' };
    case 'shells': return { text: fmtInt(v), cls: v === 1 ? 'good' : 'warn' };
    case 'bool': return { text: v ? 'yes' : 'no', cls: v ? 'good' : 'bad' };
    case 'orientation': {
      const est = m.orientation_estimated && v === 'outward' ? ' (est.)' : '';
      return { text: v + est, cls: v === 'outward' ? (est ? 'warn' : 'good') : 'bad' };
    }
    case 'volume': return { text: `${(v / 1000).toLocaleString('en-US', { maximumFractionDigits: 2 })} cm³`, cls: '' };
    case 'bbox': return { text: fmtBox(v), cls: '' };
    default: return { text: String(v), cls: '' };
  }
}

function renderMetrics(before, after) {
  const body = $('metrics').querySelector('tbody');
  body.innerHTML = '';
  for (const [label, key, kind] of ROWS) {
    const tr = document.createElement('tr');
    if (key === 'watertight') tr.className = 'key';
    const b = cell(before, key, kind);
    const a = cell(after, key, kind);
    tr.innerHTML = `<td>${label}</td><td class="${b.cls}">${escapeHtml(b.text)}</td><td class="${a.cls}">${escapeHtml(a.text || (after ? '' : '…'))}</td>`;
    body.appendChild(tr);
  }
}

function renderList(id, items, empty) {
  const ul = $(id);
  ul.innerHTML = '';
  if (!items.length) {
    ul.innerHTML = `<li class="muted">${escapeHtml(empty)}</li>`;
    return;
  }
  for (const item of items) ul.appendChild(item);
}

function renderSteps(steps) {
  renderList('changes', steps.map((s) => {
    const li = document.createElement('li');
    if (!s.changed) li.className = 'unchanged';
    li.innerHTML = `<b>${escapeHtml(s.title)}.</b> ${escapeHtml(s.summary)}`;
    return li;
  }), 'Nothing changed.');
}

function renderRemaining(items) {
  const ul = $('remaining');
  if (items === null) { ul.innerHTML = '<li class="muted">—</li>'; return; }
  if (!items.length) { ul.innerHTML = '<li class="none">Nothing found. Wall thickness and supports are not checked.</li>'; return; }
  ul.innerHTML = items.map((t) => `<li>${escapeHtml(t)}</li>`).join('');
}

function renderChecks(checks, slicer) {
  const ul = $('checks');
  if (!checks.length && !slicer) { ul.innerHTML = '<li class="muted">—</li>'; return; }
  const items = checks.map((c) => {
    const cls = c.ok === true ? 'ok' : c.ok === false ? 'fail' : 'skip';
    return `<li class="${cls}">${escapeHtml(c.label)}<span class="detail">${escapeHtml(c.detail)}</span></li>`;
  });
  if (slicer) {
    const cls = slicer.ok === true ? 'ok' : slicer.ok === false ? 'fail' : 'skip';
    const time = slicer.print_time ? ` · est. print ${slicer.print_time}` : '';
    const secs = slicer.seconds ? ` · ${slicer.seconds.toFixed(1)} s` : '';
    const who = slicer.slicer ? `Slice test (${slicer.slicer})` : 'Slice test';
    items.push(`<li class="${cls}">${escapeHtml(who)}<span class="detail">${escapeHtml(slicer.message + time + secs)}</span></li>`);
  }
  ul.innerHTML = items.join('');
}

function renderHoles(before, after, scale) {
  const table = $('holes-table');
  const rows = (before || []).slice(0, 30);
  $('holes-count').textContent = before ? `(${before.length}${before.length >= 50 ? '+' : ''} before${after ? `, ${after.length} after` : ''})` : '';
  if (!rows.length) { table.innerHTML = '<tr><td class="muted">No holes.</td></tr>'; return; }
  const cap = Number($('opt-hole-mm').value) || 0;
  const large = $('opt-large').checked;
  const head = '<tr><th>Across (mm)</th><th>Edges</th><th>Rim length (mm)</th><th>Status</th></tr>';
  const body = rows.map((h) => {
    const span = h.span_mm * scale;
    let status = large || span <= cap ? 'will be filled' : 'will stay open';
    if (after) status = h.status === 'open' ? 'still open' : 'closed';
    const cls = status === 'closed' ? 'filled' : status === 'will be filled' ? '' : 'open';
    return `<tr><td>${fmtLen(span)}</td><td>${fmtInt(h.edges)}</td><td>${fmtLen(h.perimeter_mm * scale)}</td><td class="${cls}">${status}</td></tr>`;
  }).join('');
  table.innerHTML = head + body;
}

function renderShells(shells) {
  const table = $('shells-table');
  $('shells-count').textContent = `(${shells.length})`;
  if (!shells.length) { table.innerHTML = '<tr><td class="muted">None removed.</td></tr>'; return; }
  table.innerHTML = '<tr><th>Triangles</th><th>Size (mm)</th></tr>' + shells.slice(0, 30)
    .map((s) => `<tr><td>${fmtInt(s.triangles)}</td><td>${fmtBox(s.extents)}</td></tr>`).join('');
}

// ---------------------------------------------------------------- events

function bind() {
  const drop = $('dropzone');
  const input = $('file');
  $('choose').addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => { pickFile(input.files[0]); input.value = ''; });

  let depth = 0;
  const overlay = $('drag-overlay');
  const hasFiles = (e) => Array.from(e.dataTransfer?.types || []).includes('Files');
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth += 1;
    overlay.hidden = false;
    drop.classList.add('over');
  });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } });
  window.addEventListener('dragleave', (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) { overlay.hidden = true; drop.classList.remove('over'); }
  });
  window.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    overlay.hidden = true;
    drop.classList.remove('over');
    pickFile(e.dataTransfer.files[0]);
  });

  $('unit').addEventListener('change', onUnitChange);
  $('override-unit').addEventListener('change', () => {
    const insp = state.job?.inspect;
    $('unit').disabled = !$('override-unit').checked;
    if (insp?.file_unit && !$('override-unit').checked) $('unit').value = insp.file_unit;
    onUnitChange();
  });
  for (const id of ['opt-shells', 'opt-shell-pct', 'opt-holes', 'opt-hole-mm', 'opt-large', 'opt-decimate', 'opt-target', 'opt-bed', 'opt-slice']) {
    $(id).addEventListener('change', () => {
      markStale();
      if (state.job?.inspect && state.job.state !== 'done') renderHoles(state.job.inspect.holes, null, lengthScale());
    });
  }
  $('opt-target').addEventListener('focus', () => { $('opt-decimate').checked = true; });
  $('clean').addEventListener('click', clean);
  $('show-defects').addEventListener('change', (e) => viewer.setShowDefects(e.target.checked));
  $('reset-view').addEventListener('click', () => viewer.fit());
}

bind();
init();
