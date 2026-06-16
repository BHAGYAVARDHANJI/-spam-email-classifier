/* ── Config ──────────────────────────────────────────────────────────── */
const API = 'http://localhost:5000/api';

/* ── Samples ─────────────────────────────────────────────────────────── */
const SAMPLES = {
  spam1: "URGENT! You have been selected as today's lucky winner! Click here NOW to claim your FREE iPhone 15 Pro. Limited time offer. Reply STOP to opt out.",
  spam2: "Congratulations!!! You've WON £5000 in the National Lottery. To claim your prize call 0900-XXX-XXXX immediately. Act NOW before it expires!",
  ham1:  "Hi team, just a reminder that we have a project sync meeting tomorrow at 2:00 PM in Conference Room B. Please review the agenda doc I shared earlier. Thanks!",
  ham2:  "Hey! Are you free this weekend? Thinking of catching a movie on Saturday evening. Let me know what works for you.",
};

/* ── Nav ─────────────────────────────────────────────────────────────── */
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => {
    s.classList.remove('active');
    s.classList.add('hidden');
  });
  document.querySelectorAll('.nav-pill').forEach(p => p.classList.remove('active'));

  const el = document.getElementById(`section-${name}`);
  if (el) { el.classList.remove('hidden'); el.classList.add('active'); }

  const pill = [...document.querySelectorAll('.nav-pill')]
    .find(p => p.textContent.toLowerCase().includes(name));
  if (pill) pill.classList.add('active');

  if (name === 'metrics') loadMetrics();
}

/* ── Single Classify ─────────────────────────────────────────────────── */
const emailInput = document.getElementById('emailInput');
const charCount  = document.getElementById('charCount');

emailInput && emailInput.addEventListener('input', () => {
  const n = emailInput.value.length;
  charCount.textContent = `${n} character${n !== 1 ? 's' : ''}`;
});

function setExample(key) {
  emailInput.value = SAMPLES[key] || '';
  emailInput.dispatchEvent(new Event('input'));
  document.getElementById('resultCard')?.classList.add('hidden');
}

function clearInput() {
  emailInput.value = '';
  emailInput.dispatchEvent(new Event('input'));
  document.getElementById('resultCard')?.classList.add('hidden');
}

async function classifyEmail() {
  const text = emailInput.value.trim();
  if (!text) { shake(emailInput); return; }

  const btn = document.getElementById('classifyBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Classifying…';

  try {
    const res  = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    renderResult(data);
  } catch (err) {
    alert(`Error: ${err.message}\n\nMake sure the Flask server is running on port 5000.`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">→</span> Classify';
  }
}

function renderResult(data) {
  const card    = document.getElementById('resultCard');
  const verdict = document.getElementById('resultVerdict');
  const spamBar = document.getElementById('spamBar');
  const hamBar  = document.getElementById('hamBar');
  const spamProb = document.getElementById('spamProb');
  const hamProb  = document.getElementById('hamProb');
  const preview  = document.getElementById('resultPreview');

  verdict.textContent  = data.is_spam ? '🚨 SPAM Detected' : '✅ Looks like Ham';
  verdict.className    = `result-verdict ${data.is_spam ? 'verdict-spam' : 'verdict-ham'}`;

  // Animate bars
  spamBar.style.width  = '0%';
  hamBar.style.width   = '0%';
  setTimeout(() => {
    spamBar.style.width  = `${data.spam_prob}%`;
    hamBar.style.width   = `${data.ham_prob}%`;
  }, 80);

  spamProb.textContent = `${data.spam_prob}%`;
  hamProb.textContent  = `${data.ham_prob}%`;
  preview.textContent  = `"${data.email_preview}"`;

  card.classList.remove('hidden');
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Batch Classify ──────────────────────────────────────────────────── */
async function batchClassify() {
  const raw = document.getElementById('batchInput').value.trim();
  if (!raw) return;

  const emails = raw.split('\n').map(l => l.trim()).filter(Boolean);
  if (!emails.length) return;

  try {
    const res  = await fetch(`${API}/batch-predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emails }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    renderBatch(data.results);
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

function renderBatch(results) {
  const container = document.getElementById('batchResults');
  container.innerHTML = '';
  container.classList.remove('hidden');

  results.forEach((r, i) => {
    const el = document.createElement('div');
    el.className = `batch-item ${r.is_spam ? 'is-spam' : 'is-ham'}`;
    el.style.animationDelay = `${i * 60}ms`;
    el.innerHTML = `
      <span class="batch-tag ${r.is_spam ? 'tag-spam' : 'tag-ham'}">${r.is_spam ? 'SPAM' : 'HAM'}</span>
      <span class="batch-text">${escHtml(r.email)}</span>
      <span class="batch-prob">${r.spam_prob}% spam</span>
    `;
    container.appendChild(el);
  });
}

/* ── Metrics ─────────────────────────────────────────────────────────── */
async function loadMetrics() {
  try {
    const res  = await fetch(`${API}/metrics`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    renderMetrics(data);
  } catch (err) {
    document.getElementById('metricsGrid').innerHTML =
      `<p style="color:var(--spam);grid-column:1/-1;font-size:13px;">⚠️ Could not load metrics. Is the Flask server running?</p>`;
  }
}

function renderMetrics(m) {
  const grid = document.getElementById('metricsGrid');
  const cards = [
    { label: 'Test Accuracy',  value: `${(m.test_accuracy  * 100).toFixed(1)}%`, sub: `Train: ${(m.train_accuracy * 100).toFixed(1)}%` },
    { label: 'Precision',      value: `${(m.precision      * 100).toFixed(1)}%`, sub: 'of flagged spam' },
    { label: 'Recall',         value: `${(m.recall         * 100).toFixed(1)}%`, sub: 'of actual spam caught' },
    { label: 'F1 Score',       value: `${(m.f1_score       * 100).toFixed(1)}%`, sub: 'harmonic mean' },
  ];

  grid.innerHTML = cards.map(c => `
    <div class="metric-card">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value">${c.value}</div>
      <div class="metric-sub">${c.sub}</div>
    </div>
  `).join('');

  // Confusion matrix
  const cm = m.confusion_matrix;
  document.getElementById('cmTN').textContent = cm[0][0];
  document.getElementById('cmFP').textContent = cm[0][1];
  document.getElementById('cmFN').textContent = cm[1][0];
  document.getElementById('cmTP').textContent = cm[1][1];

  // Info
  const info = [
    { key: 'Algorithm',    val: 'Multinomial NB' },
    { key: 'Dataset Size', val: m.dataset_size },
    { key: 'Train / Test', val: `${m.train_size} / ${m.test_size}` },
    { key: 'Features',     val: m.num_features.toLocaleString() },
    { key: 'Ham Emails',   val: m.class_distribution.ham ?? '—' },
    { key: 'Spam Emails',  val: m.class_distribution.spam ?? '—' },
  ];
  document.getElementById('infoGrid').innerHTML = info.map(i => `
    <div class="info-item">
      <div class="info-key">${i.key}</div>
      <div class="info-val">${i.val}</div>
    </div>
  `).join('');
}

/* ── Utils ───────────────────────────────────────────────────────────── */
function shake(el) {
  el.style.animation = 'none';
  el.offsetHeight;
  el.style.animation = 'shake .35s ease';
  setTimeout(() => { el.style.animation = ''; }, 400);
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* add shake keyframe */
const style = document.createElement('style');
style.textContent = `@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-6px)}75%{transform:translateX(6px)}}`;
document.head.appendChild(style);
