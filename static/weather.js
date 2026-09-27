(() => {
  const API = '';
  const $ = id => document.getElementById(id);
  const escapeLabel = value => String(value || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const WMO = {0:'Ciel dégagé',1:'Principalement dégagé',2:'Partiellement nuageux',3:'Couvert',45:'Brouillard',48:'Brouillard givrant',51:'Bruine légère',53:'Bruine modérée',55:'Bruine dense',61:'Pluie légère',63:'Pluie modérée',65:'Forte pluie',71:'Neige faible',80:'Averses faibles',81:'Averses modérées',82:'Fortes averses',95:'Orage'};
  const savedKey = 'prag.savedPlaces.v1';
  let places = JSON.parse(localStorage.getItem(savedKey) || '[]');
  const label = p => [p.name, p.admin1, p.country].filter(Boolean).join(', ');
  function renderSaved(selected = '') {
    const select = $('saved-places');
    select.innerHTML = '<option value="">Choisir une localité</option>' + places.map((p,i) => `<option value="${i}" ${String(i)===String(selected)?'selected':''}>${escapeLabel(label(p))}</option>`).join('');
  }
  async function search(event) {
    event.preventDefault();
    const query = $('place-query').value.trim();
    const results = $('place-results');
    results.textContent = 'Recherche…'; $('weather-message').textContent = 'Recherche de localités…';
    try {
      const response = await fetch(`${API}/api/geocode?q=${encodeURIComponent(query)}`);
      const body = await response.json(); if (!response.ok) throw new Error(body.error);
      results.replaceChildren();
      if (!body.results.length) { results.textContent = 'Aucune localité trouvée en Côte d’Ivoire.'; return; }
      body.results.forEach(place => {
        const button = document.createElement('button'); button.type = 'button'; button.setAttribute('role','option'); button.textContent = label(place);
        button.addEventListener('click', () => { results.replaceChildren(); selectPlace(place, true); }); results.append(button);
      });
      $('weather-message').textContent = 'Sélectionnez une localité dans les résultats.';
    } catch (error) { results.textContent = ''; $('weather-message').textContent = error.message || 'Recherche de localité indisponible.'; }
  }
  async function selectPlace(place, save) {
    $('weather-message').textContent = 'Chargement des prévisions météo…'; $('weather-content').classList.add('hidden');
    if (save) {
      const found = places.findIndex(p => Math.abs(p.latitude-place.latitude)<.001 && Math.abs(p.longitude-place.longitude)<.001);
      if (found >= 0) places.splice(found,1); places.unshift(place); places = places.slice(0,8);
      localStorage.setItem(savedKey, JSON.stringify(places)); renderSaved(0);
    }
    try {
      const response = await fetch(`${API}/api/weather?latitude=${encodeURIComponent(place.latitude)}&longitude=${encodeURIComponent(place.longitude)}`);
      const result = await response.json(); if (!response.ok) throw new Error(result.error);
      renderWeather(place, result); $('weather-content').classList.remove('hidden'); $('weather-message').textContent = result.stale ? 'Service météo momentanément indisponible : dernières données enregistrées affichées.' : '';
    } catch (error) { $('weather-message').textContent = error.message || 'Données météo temporairement indisponibles.'; }
  }
  function renderWeather(place, result) {
    const d = result.data, c = d.current, day = d.daily, hourly = d.hourly;
    $('weather-place').textContent = label(place); $('weather-date').textContent = new Intl.DateTimeFormat('fr-FR',{dateStyle:'full',timeZone:d.timezone}).format(new Date(c.time));
    $('weather-temp').textContent = `${Math.round(c.temperature_2m)}°`; $('weather-symbol').textContent = c.is_day ? '☀' : '☾';
    $('weather-condition').textContent = WMO[c.weather_code] || 'Conditions variables';
    $('weather-rain').textContent = `${Number(c.precipitation).toFixed(1)} mm`; $('weather-humidity').textContent = `${c.relative_humidity_2m} %`;
    $('weather-wind').textContent = `${Math.round(c.wind_speed_10m)} km/h`; $('weather-feels').textContent = `${Math.round(c.apparent_temperature)}°C`;
    const now = new Date(c.time).getTime(); let rain24=0, maxProb=0, maxTemp=-Infinity;
    for (let i=0;i<hourly.time.length;i++) if(new Date(hourly.time[i]).getTime()>=now && new Date(hourly.time[i]).getTime()<now+86400000) { rain24+=hourly.precipitation[i]||0; maxProb=Math.max(maxProb,hourly.precipitation_probability[i]||0); maxTemp=Math.max(maxTemp,hourly.temperature_2m[i]); }
    let summary = `Sur les prochaines 24 heures, ${rain24.toFixed(1)} mm de pluie sont prévus (probabilité maximale : ${maxProb} %).`;
    if (rain24>=30) summary += ' Risque de fortes précipitations : surveillez le drainage et les parcelles sensibles.';
    else if (rain24>=10) summary += ' Des pluies notables sont possibles ; adaptez les travaux au champ aux fenêtres sèches.';
    else if (rain24<2 && maxTemp>=33) summary += ' Peu de pluie et une chaleur élevée sont prévues ; vérifiez l’humidité du sol avant toute décision d’irrigation.';
    else if (rain24<2) summary += ' Peu de pluie est prévue ; surveillez l’humidité du sol selon la culture et son stade.';
    else summary += ' Les précipitations restent modérées selon les prévisions disponibles.';
    $('weather-summary').textContent = summary;
    $('weather-updated').textContent = result.stale ? `données du ${new Date(result.fetched_at*1000).toLocaleString('fr-FR')}` : result.cached ? 'depuis le cache récent' : 'à l’instant';
    const start = hourly.time.findIndex(t => new Date(t).getTime()>=now);
    $('hourly-forecast').innerHTML = Array.from({length:6},(_,n)=>{const i=start+n;if(i>=hourly.time.length)return '';return `<div class="hourly-item">${new Date(hourly.time[i]).toLocaleTimeString('fr-FR',{hour:'2-digit',timeZone:d.timezone})}<strong>${Math.round(hourly.temperature_2m[i])}°</strong>${Math.round(hourly.precipitation_probability[i]||0)}% pluie</div>`;}).join('');
    $('daily-forecast').innerHTML = day.time.slice(0,7).map((date,i)=>`<div class="daily-item"><span>${new Intl.DateTimeFormat('fr-FR',{weekday:'short',day:'numeric',month:'short',timeZone:d.timezone}).format(new Date(`${date}T12:00:00`))}</span><strong>${Math.round(day.temperature_2m_min[i])}° / ${Math.round(day.temperature_2m_max[i])}°</strong><span>${Number(day.precipitation_sum[i]).toFixed(1)} mm · ${day.precipitation_probability_max[i]||0}%</span></div>`).join('');
  }
  document.addEventListener('DOMContentLoaded', () => {
    $('place-form').addEventListener('submit', search); renderSaved();
    $('saved-places').addEventListener('change', e => { if(e.target.value!=='') selectPlace(places[Number(e.target.value)],false); });
    if (places.length) selectPlace(places[0],false);
  });
})();
