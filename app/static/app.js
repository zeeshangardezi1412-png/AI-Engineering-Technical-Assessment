'use strict';

const ALLOWED = ['.wav', '.mp3', '.m4a', '.flac', '.ogg'];
const KEY_STORAGE = 'transcription.apiKey';
const POLL_MS = 1000;
const STEPS = ['upload', 'queued', 'processing', 'completed'];

const $ = (id) => document.getElementById(id);
const els = {
  key: $('api-key'), toggleKey: $('toggle-key'), file: $('file'), drop: $('drop'),
  dropMain: $('drop-main'), dropSub: $('drop-sub'), submit: $('submit'), reset: $('reset'),
  job: $('job'), jobId: $('job-id'), steps: $('steps'), bar: $('bar'), barFill: $('bar-fill'),
  status: $('status-text'), error: $('error'), result: $('result'), meta: $('meta'),
  player: $('player'), segments: $('segments'), copy: $('copy'),
  srt: $('download-srt'), json: $('download-json'), health: $('health'), healthText: $('health-text'),
};

let selectedFile = null;
let busy = false;
let pollTimer = null;
let result = null;
let audioUrl = null;
let activeIndex = -1;

// --- storage (per-tab convenience only; never required) ------------------------
function loadKey() {
  try { return sessionStorage.getItem(KEY_STORAGE) || ''; } catch { return ''; }
}
function saveKey(value) {
  try { sessionStorage.setItem(KEY_STORAGE, value); } catch { /* storage unavailable */ }
}

// --- formatting ----------------------------------------------------------------
function clock(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds - m * 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}
function srtTime(seconds) {
  const ms = Math.round(seconds * 1000);
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor(ms / 60000) % 60)}:${pad(Math.floor(ms / 1000) % 60)},${pad(ms % 1000, 3)}`;
}
function fileSize(bytes) {
  return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
function extension(name) {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot).toLowerCase() : '';
}

// --- UI state ------------------------------------------------------------------
function updateSubmit() {
  els.submit.disabled = busy || !selectedFile || !els.key.value.trim();
}
function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}
function clearError() {
  els.error.hidden = true;
  els.error.textContent = '';
}
function setStep(step, failed = false) {
  const current = STEPS.indexOf(step);
  for (const li of els.steps.children) {
    const index = STEPS.indexOf(li.dataset.step);
    li.className = index < current ? 'done' : index === current ? (failed ? 'failed' : 'active') : '';
  }
  if (step === 'completed' && !failed) els.steps.lastElementChild.className = 'done';
}
function setProgress(percent) {
  if (percent == null) {
    els.bar.classList.add('indeterminate');
    els.bar.removeAttribute('aria-valuenow');
    els.barFill.style.width = '';
  } else {
    const value = Math.max(0, Math.min(100, percent));
    els.bar.classList.remove('indeterminate');
    els.bar.setAttribute('aria-valuenow', String(Math.round(value)));
    els.barFill.style.width = `${value}%`;
  }
}
function setStatus(text) {
  els.status.textContent = text;
}

function selectFile(file) {
  clearError();
  if (!file) return;
  if (!ALLOWED.includes(extension(file.name))) {
    showError(`“${file.name}” is not a supported format. Use WAV, MP3, M4A, FLAC or OGG.`);
    return;
  }
  selectedFile = file;
  els.dropMain.textContent = file.name;
  els.dropSub.textContent = `${fileSize(file.size)} · click to choose another file`;
  els.drop.classList.add('has-file');
  updateSubmit();
}

function resetAll() {
  clearTimeout(pollTimer);
  pollTimer = null;
  busy = false;
  selectedFile = null;
  result = null;
  activeIndex = -1;
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = null;
  els.player.removeAttribute('src');
  els.player.load();
  els.file.value = '';
  els.dropMain.innerHTML = 'Drop an audio file here or <u>browse</u>';
  els.dropSub.textContent = 'WAV, MP3, M4A, FLAC or OGG';
  els.drop.classList.remove('has-file');
  els.job.hidden = true;
  els.result.hidden = true;
  els.reset.hidden = true;
  els.segments.replaceChildren();
  clearError();
  updateSubmit();
}

function finish() {
  busy = false;
  els.reset.hidden = false;
  updateSubmit();
}

// --- API -----------------------------------------------------------------------
function headers() {
  return { 'X-API-Key': els.key.value.trim() };
}

async function errorMessage(response) {
  let detail = '';
  try {
    const text = await response.text();
    try { detail = JSON.parse(text).detail || text; } catch { detail = text; }
  } catch { /* no body */ }
  if (response.status === 401) return 'The API key was rejected. Check the value of API_KEY.';
  if (response.status === 413) return detail && detail !== 'Content Too Large' ? detail : 'The file is larger than the upload limit.';
  if (response.status === 503) return 'The service is temporarily unavailable. Try again in a moment.';
  return detail || `Request failed (HTTP ${response.status}).`;
}

function upload() {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', selectedFile);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/transcriptions');
    xhr.setRequestHeader('X-API-Key', els.key.value.trim());
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = (event.loaded / event.total) * 100;
      setProgress(percent);
      setStatus(percent < 100 ? `Uploading… ${Math.round(percent)}%` : 'Checking the audio…');
    };
    xhr.onload = () => resolve(new Response(xhr.responseText, { status: xhr.status }));
    xhr.onerror = () => reject(new Error('Could not reach the service.'));
    xhr.send(form);
  });
}

async function start() {
  if (!selectedFile) return;
  clearError();
  busy = true;
  updateSubmit();
  els.result.hidden = true;
  els.job.hidden = false;
  els.jobId.textContent = '';
  setStep('upload');
  setProgress(0);
  setStatus('Uploading…');
  try {
    const response = await upload();
    if (response.status !== 202) {
      setStep('upload', true);
      setProgress(0);
      setStatus('Upload rejected.');
      showError(await errorMessage(response));
      finish();
      return;
    }
    const { job_id: jobId } = await response.json();
    els.jobId.textContent = jobId;
    setStep('queued');
    setProgress(null);
    setStatus('Waiting for a worker…');
    poll(jobId);
  } catch (error) {
    setStep('upload', true);
    setStatus('Upload failed.');
    showError(error.message);
    finish();
  }
}

async function poll(jobId) {
  let response;
  try {
    response = await fetch(`/transcriptions/${encodeURIComponent(jobId)}`, { headers: headers() });
  } catch {
    setStatus('Connection lost. Retrying…');
    pollTimer = setTimeout(() => poll(jobId), POLL_MS * 3);
    return;
  }
  if (response.status === 503) {
    setStatus('The service is temporarily unavailable. Retrying…');
    pollTimer = setTimeout(() => poll(jobId), POLL_MS * 3);
    return;
  }
  if (!response.ok) {
    showError(response.status === 404 ? 'This job no longer exists (it may have expired).' : await errorMessage(response));
    finish();
    return;
  }

  const job = await response.json();
  if (job.status === 'pending') {
    setStep('queued');
    setProgress(null);
    setStatus('Waiting for a worker…');
  } else if (job.status === 'processing') {
    setStep('processing');
    const p = job.progress;
    if (p && p.duration_seconds) {
      setProgress((p.processed_seconds / p.duration_seconds) * 100);
      setStatus(`Transcribed ${Math.round(p.processed_seconds)} s of ${Math.round(p.duration_seconds)} s · chunk ${p.completed_chunks} of ~${p.total_chunks}`);
    } else {
      setProgress(null);
      setStatus('Transcribing… (the first job also loads the Whisper model)');
    }
  } else if (job.status === 'retrying') {
    setStep('processing');
    setProgress(null);
    setStatus('A temporary error occurred. Retrying automatically…');
  } else if (job.status === 'completed') {
    setStep('completed');
    setProgress(100);
    setStatus('Completed.');
    await loadResult(jobId);
    finish();
    return;
  } else if (job.status === 'failed') {
    setStep('processing', true);
    setProgress(0);
    setStatus('Failed.');
    showError(job.error || 'Transcription failed.');
    finish();
    return;
  }
  pollTimer = setTimeout(() => poll(jobId), POLL_MS);
}

async function loadResult(jobId) {
  const response = await fetch(`/transcriptions/${encodeURIComponent(jobId)}/result`, { headers: headers() });
  if (!response.ok) {
    showError(await errorMessage(response));
    return;
  }
  result = await response.json();
  renderResult();
}

// --- result view -----------------------------------------------------------------
function renderResult() {
  const segments = result.segments || [];
  const end = segments.length ? segments[segments.length - 1].end : 0;
  els.meta.textContent = [
    result.filename,
    result.language ? `language: ${result.language}` : null,
    `${segments.length} segment${segments.length === 1 ? '' : 's'}`,
    clock(end),
  ].filter(Boolean).join(' · ');

  if (audioUrl) URL.revokeObjectURL(audioUrl);
  audioUrl = URL.createObjectURL(selectedFile);
  els.player.src = audioUrl;

  const items = segments.map((segment, index) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    const time = document.createElement('time');
    time.textContent = `${clock(segment.start)} – ${clock(segment.end)}`;
    const text = document.createElement('span');
    text.textContent = segment.text;  // never innerHTML: transcript text is untrusted
    button.append(time, text);
    button.addEventListener('click', () => {
      els.player.currentTime = segment.start;
      els.player.play().catch(() => {});
      highlight(index);
    });
    li.append(button);
    return li;
  });
  els.segments.replaceChildren(...items);
  if (!items.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No speech was detected.';
    els.segments.append(li);
  }
  activeIndex = -1;
  els.result.hidden = false;
}

function highlight(index) {
  if (index === activeIndex) return;
  const items = els.segments.children;
  if (activeIndex >= 0 && items[activeIndex]) items[activeIndex].classList.remove('active');
  activeIndex = index;
  if (index >= 0 && items[index]) {
    items[index].classList.add('active');
    const list = els.segments;
    const item = items[index];
    if (item.offsetTop < list.scrollTop || item.offsetTop + item.offsetHeight > list.scrollTop + list.clientHeight) {
      list.scrollTo({ top: item.offsetTop - list.offsetTop - 8, behavior: 'smooth' });
    }
  }
}

function onTimeUpdate() {
  if (!result) return;
  const t = els.player.currentTime;
  const index = (result.segments || []).findIndex((s) => t >= s.start && t < s.end);
  highlight(index);
}

function download(name, type, content) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function baseName() {
  const name = (result && result.filename) || 'transcript';
  return name.replace(/\.[^.]+$/, '') || 'transcript';
}

function toSrt(segments) {
  return segments.map((s, i) => `${i + 1}\n${srtTime(s.start)} --> ${srtTime(s.end)}\n${s.text}\n`).join('\n');
}

function legacyCopy(text) {
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.append(area);
  area.select();
  const ok = document.execCommand('copy');
  area.remove();
  return ok;
}

async function copyText() {
  const text = result.transcription || '';
  let ok = false;
  try {
    // The Clipboard API exists only on HTTPS or localhost; plain-HTTP LAN access uses the fallback.
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      ok = true;
    } else {
      ok = legacyCopy(text);
    }
  } catch {
    ok = legacyCopy(text);
  }
  els.copy.textContent = ok ? 'Copied' : 'Copy failed';
  setTimeout(() => { els.copy.textContent = 'Copy text'; }, 1500);
}

async function checkHealth() {
  try {
    const response = await fetch('/health');
    const up = response.ok;
    els.health.dataset.state = up ? 'up' : 'down';
    els.healthText.textContent = up ? 'Service online' : 'Service unavailable';
  } catch {
    els.health.dataset.state = 'down';
    els.healthText.textContent = 'Service unreachable';
  }
}

// --- wiring --------------------------------------------------------------------------
els.key.value = loadKey();
els.key.addEventListener('input', () => { saveKey(els.key.value.trim()); updateSubmit(); });
els.toggleKey.addEventListener('click', () => {
  const show = els.key.type === 'password';
  els.key.type = show ? 'text' : 'password';
  els.toggleKey.textContent = show ? 'Hide' : 'Show';
  els.toggleKey.setAttribute('aria-pressed', String(show));
});
els.file.addEventListener('change', () => selectFile(els.file.files[0]));
for (const type of ['dragenter', 'dragover']) {
  els.drop.addEventListener(type, (event) => { event.preventDefault(); els.drop.classList.add('over'); });
}
for (const type of ['dragleave', 'drop']) {
  els.drop.addEventListener(type, () => els.drop.classList.remove('over'));
}
els.drop.addEventListener('drop', (event) => {
  event.preventDefault();
  if (!busy) selectFile(event.dataTransfer.files[0]);
});
els.submit.addEventListener('click', start);
els.reset.addEventListener('click', resetAll);
els.player.addEventListener('timeupdate', onTimeUpdate);
els.copy.addEventListener('click', copyText);
els.srt.addEventListener('click', () => download(`${baseName()}.srt`, 'application/x-subrip', toSrt(result.segments || [])));
els.json.addEventListener('click', () => download(`${baseName()}.json`, 'application/json', JSON.stringify(result, null, 2)));

updateSubmit();
checkHealth();
