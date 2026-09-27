/* ═══════════════════════════════════════════════════════════
   PRAG — Application JavaScript
   Gestion des prédictions, affichage, historique
   ═══════════════════════════════════════════════════════════ */

const API = window.location.origin;

const MOIS_NOMS = ['', 'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                   'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'];

// ────────────────────────────────────────────────────────────
// INIT
// ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  loadHistory();
  setupForm();
  animateOnScroll();
});

// ────────────────────────────────────────────────────────────
// API HEALTH CHECK
// ────────────────────────────────────────────────────────────
async function checkApiHealth() {
  const badge = document.getElementById('model-status');
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    if (data.models_loaded) {
      badge.textContent = ' Modèles actifs';
      badge.className = 'status-badge status-ok';
    } else {
      badge.textContent = ' Modèles non chargés';
      badge.className = 'status-badge status-loading';
    }
  } catch (e) {
    badge.textContent = ' API hors ligne';
    badge.className = 'status-badge status-error';
  }
}

// ────────────────────────────────────────────────────────────
// CLIMAT PRESETS & FORM SETUP
// ────────────────────────────────────────────────────────────
const CLIMAT_PRESETS = {
  'Haut-Sassandra': {
    pluies: [5, 6, 7, 9, 10], trans: [4, 8],
    pluv_p: 190, temp_p: 26.5, hum_p: 86,
    pluv_t: 130, temp_t: 27.5, hum_t: 78,
    pluv_s: 45,  temp_s: 30.0, hum_s: 68
  },
  'Nawa': {
    pluies: [5, 6, 7, 9, 10], trans: [4, 8],
    pluv_p: 195, temp_p: 26.0, hum_p: 88,
    pluv_t: 135, temp_t: 27.0, hum_t: 80,
    pluv_s: 50,  temp_s: 29.5, hum_s: 70
  },
  'Poro': {
    pluies: [6, 7, 8, 9], trans: [5, 10],
    pluv_p: 165, temp_p: 28.0, hum_p: 76,
    pluv_t: 90,  temp_t: 30.0, hum_t: 60,
    pluv_s: 15,  temp_s: 33.5, hum_s: 42
  }
};

function autoSuggestClimate() {
  const reg = document.getElementById('region').value;
  const m = parseInt(document.getElementById('mois').value);
  if (!reg || !m || !CLIMAT_PRESETS[reg]) return;

  const ref = CLIMAT_PRESETS[reg];
  let p, t, h;
  if (ref.pluies.includes(m)) {
    p = ref.pluv_p; t = ref.temp_p; h = ref.hum_p;
  } else if (ref.trans.includes(m)) {
    p = ref.pluv_t; t = ref.temp_t; h = ref.hum_t;
  } else {
    p = ref.pluv_s; t = ref.temp_s; h = ref.hum_s;
  }

  const pluvInput = document.getElementById('pluviometrie_mm');
  const tempInput = document.getElementById('temperature_celsius');
  const humInput  = document.getElementById('humidite_pct');

  if (pluvInput) {
    pluvInput.value = p;
    document.getElementById('pluv-val').textContent = `${p} mm`;
  }
  if (tempInput) {
    tempInput.value = t;
    document.getElementById('temp-val').textContent = `${t}°C`;
  }
  if (humInput) {
    humInput.value = h;
    document.getElementById('hum-val').textContent = `${h}%`;
  }
}

function setupForm() {
  const form = document.getElementById('predict-form');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await runPrediction();
  });

  const regSelect = document.getElementById('region');
  const moisSelect = document.getElementById('mois');
  if (regSelect) regSelect.addEventListener('change', autoSuggestClimate);
  if (moisSelect) moisSelect.addEventListener('change', autoSuggestClimate);
}

// ────────────────────────────────────────────────────────────
// PREDICTION
// ────────────────────────────────────────────────────────────
async function runPrediction() {
  const btn = document.getElementById('predict-btn');
  const loader = btn.querySelector('.btn-loader');
  const btnText = btn.querySelector('.btn-text');
  const btnIcon = btn.querySelector('.btn-icon');

  // Show loading state
  btn.disabled = true;
  loader.classList.remove('hidden');
  btnText.textContent = 'Analyse en cours...';
  btnIcon.classList.add('hidden');

  try {
    const payload = collectFormData();
    const res = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Erreur serveur');
    }

    displayResults(data, payload);
    loadHistory(); // Refresh history

  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    loader.classList.add('hidden');
    btnText.textContent = 'Analyser ma période';
    btnIcon.classList.remove('hidden');
  }
}

function collectFormData() {
  return {
    region:                document.getElementById('region').value,
    culture:               document.getElementById('culture').value,
    type_sol:              document.getElementById('type_sol').value,
    mois:                  parseInt(document.getElementById('mois').value),
    annee:                 new Date().getFullYear(),
    pluviometrie_mm:       parseFloat(document.getElementById('pluviometrie_mm').value),
    temperature_celsius:   parseFloat(document.getElementById('temperature_celsius').value),
    humidite_pct:          parseFloat(document.getElementById('humidite_pct').value),
    jour_annee:            Math.round(parseInt(document.getElementById('mois').value) * 30.4)
  };
}

// ────────────────────────────────────────────────────────────
// DISPLAY RESULTS
// ────────────────────────────────────────────────────────────
function displayResults(data, payload) {
  const panel = document.getElementById('results-panel');
  panel.classList.remove('hidden');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const pred    = data.prediction;
  const rapport = data.rapport_consequences;
  const isOk    = pred.periode_semis_optimale === 1;
  const proba   = Math.round(pred.probabilite_optimal * 100);

  // ── Verdict Card ────────────────────────────────────────
  const verdictCard  = document.getElementById('verdict-card');
  const verdictIcon  = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictSub   = document.getElementById('verdict-sub');

  verdictCard.className = `verdict-card glass-card ${isOk ? 'optimal' : 'suboptimal'}`;

  if (isOk) {
    verdictIcon.textContent = '';
    verdictTitle.textContent = 'Période favorable !';
    verdictTitle.style.color = '#e97827';
  } else {
    verdictIcon.textContent = '';
    verdictTitle.textContent = 'Période non optimale';
    verdictTitle.style.color = '#d98779';
  }

  verdictSub.textContent = `${payload.culture} — ${payload.region} — ${MOIS_NOMS[payload.mois]}`;

  // Probability ring animation
  animateRing(proba, isOk);

  // Metrics
  document.getElementById('rendement-val').textContent = pred.rendement_predit_tonnes_ha.toFixed(2);

  const riskEl = document.getElementById('risk-val');
  riskEl.textContent = rapport.niveau_risque;
  riskEl.className = `metric-val risk-${rapport.niveau_risque === 'FAIBLE' ? 'low' : rapport.niveau_risque === 'MODÉRÉ' ? 'medium' : 'high'}`;

  document.getElementById('best-month-val').textContent = rapport.meilleur_mois?.nom_mois || '—';

  // ── Consequences ────────────────────────────────────────
  renderConsequences(rapport);

  // ── Monthly Calendar ────────────────────────────────────
  const calCard = document.getElementById('calendar-card');
  if (calCard) calCard.classList.remove('hidden');
  renderMonthlyCalendar(rapport.analyse_annuelle, payload.mois, payload.culture);
}

function animateRing(proba, isOk) {
  const arc = document.getElementById('ring-arc');
  const val = document.getElementById('proba-val');

  // Color based on proba
  const color = proba >= 60 ? '#e97827' : proba >= 40 ? '#d99a60' : '#d98779';
  arc.style.stroke = color;

  // Animate circumference
  const circumference = 2 * Math.PI * 15.9;
  let current = 0;
  const target = (proba / 100) * 100;

  arc.style.strokeDasharray = `0 100`;
  val.textContent = '0';

  const duration = 1000;
  const startTime = performance.now();

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const currentVal = Math.round(target * eased);

    arc.style.strokeDasharray = `${target * eased} 100`;
    val.textContent = currentVal;

    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

function renderConsequences(rapport) {
  const list = document.getElementById('consequences-list');
  list.innerHTML = '';

  rapport.consequences.forEach((c, i) => {
    const div = document.createElement('div');
    const isOk = c.startsWith('');
    const isWarn = c.startsWith('') || c.startsWith('') || c.startsWith('') || c.startsWith('');
    const isDanger = c.startsWith('');

    div.className = `consequence-item ${isDanger ? 'consequence-error' : isOk ? 'consequence-ok' : 'consequence-warn'}`;
    div.style.animationDelay = `${i * 80}ms`;
    div.textContent = c;
    list.appendChild(div);
  });

  // Mois optimaux
  const moisSection = document.getElementById('mois-optimaux-section');
  const moisBadges  = document.getElementById('mois-badges');
  moisBadges.innerHTML = '';

  if (rapport.mois_optimaux && rapport.mois_optimaux.length > 0) {
    moisSection.classList.remove('hidden');
    rapport.mois_optimaux.forEach(mois => {
      const badge = document.createElement('span');
      badge.className = 'mois-badge';
      badge.textContent = mois;
      moisBadges.appendChild(badge);
    });
  } else {
    moisSection.classList.add('hidden');
  }
}

function renderMonthlyCalendar(analyse, currentMois, culture) {
  const grid = document.getElementById('monthly-grid');
  const cultureNameEl = document.getElementById('cal-culture-name');
  cultureNameEl.textContent = culture;

  const MOIS_SHORT = ['', 'Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                      'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'];

  grid.innerHTML = '';

  const maxRend = Math.max(...analyse.map(m => m.rendement_predit));

  analyse.forEach(m => {
    const cell = document.createElement('div');
    cell.className = `month-cell ${m.est_optimal ? 'optimal' : ''} ${m.mois === currentMois ? 'current' : ''}`;
    cell.title = `${m.nom_mois}\nProbabilité : ${Math.round(m.probabilite_optimal * 100)}%\nRendement : ${m.rendement_predit} t/ha`;

    const barWidth = (m.probabilite_optimal * 100).toFixed(0);

    cell.innerHTML = `
      <div class="month-name">${MOIS_SHORT[m.mois]}</div>
      <div class="month-proba">${Math.round(m.probabilite_optimal * 100)}%</div>
      <div class="month-rend">${m.rendement_predit}t</div>
      <div class="month-bar" style="width: ${barWidth}%"></div>
    `;
    grid.appendChild(cell);
  });
}

// ────────────────────────────────────────────────────────────
// HISTORY
// ────────────────────────────────────────────────────────────
async function loadHistory() {
  const container = document.getElementById('history-container');

  try {
    const res = await fetch(`${API}/api/history`);
    const data = await res.json();

    const sessions = data.sessions || [];
    const predictions = data.predictions || [];

    if (sessions.length === 0 && predictions.length === 0) {
      container.innerHTML = `
        <div class="history-empty">
          <p> Aucun historique disponible. Lancez d'abord l'entraînement et effectuez des prédictions.</p>
        </div>
      `;
      return;
    }

    let html = '';

    // Sessions d'entraînement
    if (sessions.length > 0) {
      html += `<h3 style="color: var(--text-muted); font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.75rem">
         Sessions d'entraînement (${sessions.length})
      </h3>`;

      sessions.slice().reverse().forEach(s => {
        const date = new Date(s.started_at).toLocaleString('fr-FR');
        const clfOp = s.operations?.find(o => o.op === 'classification_training');
        const regOp = s.operations?.find(o => o.op === 'regression_training');

        html += `
          <div class="history-session">
            <div class="session-header">
              <span class="session-id">Session : ${date}</span>
              <span class="session-badge badge-${s.status === 'completed' ? 'completed' : 'running'}">
                ${s.status === 'completed' ? ' Terminé' : '⏳ En cours'}
              </span>
            </div>
            <div class="session-metrics">
              ${clfOp ? `<span class="session-metric"> CLF : <strong>${clfOp.best_model}</strong> | F1=<strong>${clfOp.best_metrics?.f1_score}</strong></span>` : ''}
              ${regOp ? `<span class="session-metric"> REG : <strong>${regOp.best_model}</strong> | RMSE=<strong>${regOp.best_metrics?.rmse}</strong></span>` : ''}
              ${s.total_duration_sec ? `<span class="session-metric">⏱ Durée : <strong>${s.total_duration_sec}s</strong></span>` : ''}
            </div>
          </div>
        `;
      });
    }

    // Prédictions récentes
    if (predictions.length > 0) {
      html += `<h3 style="color: var(--text-muted); font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin:1.5rem 0 0.75rem">
         Prédictions récentes (${predictions.length})
      </h3>`;

      predictions.slice(-10).reverse().forEach(p => {
        const date = new Date(p.timestamp).toLocaleString('fr-FR');
        const isOk = p.output?.periode_semis_optimale === 1;
        html += `
          <div class="history-session">
            <div class="session-header">
              <span class="session-id">${date} — ${p.input?.culture} / ${p.input?.region} / ${MOIS_NOMS[p.input?.mois]}</span>
              <span class="session-badge ${isOk ? 'badge-completed' : 'badge-error'}">
                ${isOk ? ' Optimal' : ' Non optimal'}
              </span>
            </div>
            <div class="session-metrics">
              <span class="session-metric"> Rendement : <strong>${p.output?.rendement_predit_tonnes_ha} t/ha</strong></span>
              <span class="session-metric"> Probabilité : <strong>${Math.round((p.output?.probabilite_optimal || 0) * 100)}%</strong></span>
              <span class="session-metric"> Risque : <strong>${p.risk_level}</strong></span>
            </div>
          </div>
        `;
      });
    }

    // Best models
    if (data.best_models && (data.best_models.classification || data.best_models.regression)) {
      html += `<h3 style="color: var(--text-muted); font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin:1.5rem 0 0.75rem">
         Meilleurs modèles actuels
      </h3>
      <div class="history-session">
        <div class="session-metrics" style="gap:2rem">
          ${data.best_models.classification ? `
            <span class="session-metric">
               Classification : <strong>${data.best_models.classification.model}</strong>
              | F1=<strong>${data.best_models.classification.metrics?.f1_score}</strong>
              | AUC=<strong>${data.best_models.classification.metrics?.roc_auc}</strong>
              | Inférence=<strong>${data.best_models.classification.metrics?.inference_ms_per_obs}ms</strong>
            </span>
          ` : ''}
          ${data.best_models.regression ? `
            <span class="session-metric">
               Régression : <strong>${data.best_models.regression.model}</strong>
              | RMSE=<strong>${data.best_models.regression.metrics?.rmse}</strong>
              | R²=<strong>${data.best_models.regression.metrics?.r2_score}</strong>
            </span>
          ` : ''}
        </div>
      </div>`;
    }

    container.innerHTML = html;

  } catch (e) {
    container.innerHTML = `
      <div class="history-empty">
        <p> Impossible de charger l'historique. L'API est-elle démarrée ?</p>
      </div>
    `;
  }
}

// ────────────────────────────────────────────────────────────
// ERROR DISPLAY
// ────────────────────────────────────────────────────────────
function showError(message) {
  const panel = document.getElementById('results-panel');
  panel.classList.remove('hidden');

  const verdict = document.getElementById('verdict-card');
  verdict.className = 'verdict-card glass-card suboptimal';

  document.getElementById('verdict-icon').textContent = '';
  document.getElementById('verdict-title').textContent = 'Erreur';
  document.getElementById('verdict-title').style.color = '#d98779';
  document.getElementById('verdict-sub').textContent = message;

  document.getElementById('consequences-list').innerHTML = `
    <div class="consequence-item consequence-error">
      ${message.includes('Modèles non chargés')
        ? ' Les modèles IA ne sont pas encore chargés. Lancez <code>python prag_train.py</code> pour les entraîner.'
        : ` ${message}`
      }
    </div>
  `;
  document.getElementById('mois-optimaux-section').classList.add('hidden');
  document.getElementById('calendar-card').classList.add('hidden');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ────────────────────────────────────────────────────────────
// ANIMATIONS ON SCROLL
// ────────────────────────────────────────────────────────────
function animateOnScroll() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll('.glass-card, .stat-card, .section-header').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });
}
