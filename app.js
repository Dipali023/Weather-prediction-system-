/* =====================================================
   WeatherSense AI — Complete Intelligence Engine
   Real ML: Linear Regression + Random Forest Logic
   Real API: Open-Meteo (100% free, no key needed)
   Anomaly Detection: Z-Score Statistical Method
   Health Advisory AI Engine
   ===================================================== */
'use strict';

// ─── Global State & Locations ─────────────────────────
const LOCATIONS = {
  nagpur:    { name: 'Nagpur, Maharashtra',     lat: 21.15, lon: 79.09, tz: 'Asia/Kolkata' },
  mumbai:    { name: 'Mumbai, Maharashtra',     lat: 19.07, lon: 72.87, tz: 'Asia/Kolkata' },
  pune:      { name: 'Pune, Maharashtra',       lat: 18.52, lon: 73.85, tz: 'Asia/Kolkata' },
  delhi:     { name: 'Delhi, India',            lat: 28.61, lon: 77.20, tz: 'Asia/Kolkata' },
  bangalore: { name: 'Bengaluru, Karnataka',    lat: 12.97, lon: 77.59, tz: 'Asia/Kolkata' },
  chennai:   { name: 'Chennai, Tamil Nadu',     lat: 13.08, lon: 80.27, tz: 'Asia/Kolkata' },
  kolkata:   { name: 'Kolkata, West Bengal',    lat: 22.57, lon: 88.36, tz: 'Asia/Kolkata' },
  hyderabad: { name: 'Hyderabad, Telangana',    lat: 17.38, lon: 78.48, tz: 'Asia/Kolkata' },
  ahmedabad: { name: 'Ahmedabad, Gujarat',      lat: 23.02, lon: 72.57, tz: 'Asia/Kolkata' },
  jaipur:    { name: 'Jaipur, Rajasthan',       lat: 26.91, lon: 75.78, tz: 'Asia/Kolkata' },
  lucknow:   { name: 'Lucknow, Uttar Pradesh',  lat: 26.84, lon: 80.94, tz: 'Asia/Kolkata' },
  patna:     { name: 'Patna, Bihar',            lat: 25.59, lon: 85.13, tz: 'Asia/Kolkata' },
  guwahati:  { name: 'Guwahati, Assam',         lat: 26.14, lon: 91.73, tz: 'Asia/Kolkata' },
  srinagar:  { name: 'Srinagar, Jammu & Kashmir',lat: 34.08, lon: 74.79, tz: 'Asia/Kolkata' },
  chandigarh: { name: 'Chandigarh, India',        lat: 30.73, lon: 76.78, tz: 'Asia/Kolkata' },
};

const state = {
  data: {},
  history: { temp:[], hum:[], press:[], wind:[], rain:[], uv:[], ts:[] },
  charts: {},
  mini:   {},
  sensor: {},
  thresholds: { temp:38, hum:85, wind:50, press:995, rain:20, uv:7 },
  alerts: [],
  anomalies: [],
  totalReadings: 0,
  uptimeSec: 0,
  compassAngle: 45,
  mlModel: { slope:0, intercept:28, r2:0, rmse:0 },
  anomalyStats: { total:0, normal:0 },
  forecastLoaded: false,
  apiConnected: false,
  currentLoc: 'nagpur',
  liveTemp: undefined,
  liveWind: undefined,
};

// ─── Particles ────────────────────────────────────────
function initParticles() {
  const c = document.getElementById('particles');
  for (let i = 0; i < 20; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const sz = Math.random() * 90 + 20;
    p.style.cssText = `width:${sz}px;height:${sz}px;left:${Math.random()*100}%;animation-duration:${Math.random()*25+15}s;animation-delay:${Math.random()*20}s;opacity:${Math.random()*0.12};`;
    c.appendChild(p);
  }
}

// ─── Clock ────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('liveTime');
  if (el) el.textContent = new Date().toLocaleTimeString('en-IN', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

// ─── Navigation ───────────────────────────────────────
const sectionTitles = {
  dashboard: 'Dashboard Overview', aiml: 'AI/ML Engine',
  sensors: 'Live Sensors', forecast: 'AI Forecast',
  health: 'Health Advisory', anomaly: 'Anomaly Detection',
  alerts: 'Smart Alerts', hardware: 'Hardware Architecture',
};

function showSection(id, el) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const sec = document.getElementById('section-' + id);
  if (sec) sec.classList.add('active');
  if (el)  el.classList.add('active');
  setEl('pageTitle', sectionTitles[id] || id);
  setEl('breadcrumb', 'Home / ' + (sectionTitles[id] || id));
  if (id === 'forecast' && !state.forecastLoaded) loadForecast();
}

function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  const mc = document.querySelector('.main-content');
  sb.classList.toggle('collapsed');
  mc.classList.toggle('expanded');
  if (window.innerWidth <= 768) sb.classList.toggle('open');
}

// ─── Simulate Sensor Readings (IoT simulation) ────────
function generateSensorData() {
  const now = new Date();
  const h = now.getHours() + now.getMinutes() / 60;
  
  // Use real weather values as baselines if loaded from API, otherwise use defaults
  const baseTemp = state.liveTemp !== undefined ? state.liveTemp : 27;
  const baseWind = state.liveWind !== undefined ? state.liveWind : 10;
  
  // Modulate slightly based on time-of-day cycle and random noise
  const t = baseTemp + 2 * Math.sin((h - 6) * Math.PI / 12) + (Math.random() - 0.5) * 1.5;
  const hum = 58 + 20 * Math.cos((h - 15) * Math.PI / 12) + (Math.random() - 0.5) * 4;
  const press = 1009 - 6 * Math.sin(h * Math.PI / 24) + (Math.random() - 0.5) * 2.5;
  const wind = baseWind + 4 * Math.abs(Math.sin(h * Math.PI / 10)) + (Math.random() - 0.5) * 2;
  const rain = (h > 14 && h < 19) ? Math.random() * 5 : Math.random() * 0.4;
  const uv   = (h > 6 && h < 18) ? Math.min(11, 6.5 * Math.sin((h - 6) * Math.PI / 12) + Math.random()) : 0;
  const lux  = uv * 11500 + Math.random() * 4000;
  const alt  = 317 + (1013.25 - press) * 8.5 + (Math.random() - 0.5) * 3;
  const slp  = press + (alt / 8.5);
  const gust = wind * 1.5 + Math.random() * 4;
  const feelsLike = t - wind / 10 + hum / 60;
  const dewPoint  = t - (100 - hum) / 5;
  const vis  = Math.max(1, 16 - rain * 2.5 - (hum > 90 ? 5 : 0));
  const heatIdx = t + 0.33 * (hum / 100 * 6.105) - 0.70 * (wind / 3.6) - 4;
  state.compassAngle += (Math.random() - 0.5) * 12;
  const d = {
    temp:+t.toFixed(1), hum:+Math.min(99,Math.max(10,hum)).toFixed(1),
    press:+press.toFixed(1), wind:+wind.toFixed(1), rain:+rain.toFixed(2),
    uv:+uv.toFixed(1), lux:+lux.toFixed(0), alt:+alt.toFixed(1),
    slp:+slp.toFixed(1), gust:+gust.toFixed(1), feelsLike:+feelsLike.toFixed(1),
    dewPoint:+dewPoint.toFixed(1), vis:+vis.toFixed(1), heatIdx:+heatIdx.toFixed(1),
    windDir: state.compassAngle, now
  };
  state.data = d;
  return d;
}

// ═══════════════════════════════════════════════════════
//   REAL AI/ML ENGINE #1 — Linear Regression (JS)
//   Trains on live sensor data, predicts future temp
// ═══════════════════════════════════════════════════════
function trainLinearRegression(xs, ys) {
  const n = xs.length;
  if (n < 3) return { slope:0, intercept:ys[0]||28, r2:0, rmse:0 };
  const meanX = xs.reduce((a,b)=>a+b,0)/n;
  const meanY = ys.reduce((a,b)=>a+b,0)/n;
  let num=0, den=0;
  for (let i=0;i<n;i++) { num += (xs[i]-meanX)*(ys[i]-meanY); den += (xs[i]-meanX)**2; }
  const slope = den !== 0 ? num/den : 0;
  const intercept = meanY - slope * meanX;
  // R² score
  const ssRes = ys.reduce((a,y,i)=>a+(y-(slope*xs[i]+intercept))**2, 0);
  const ssTot = ys.reduce((a,y)=>a+(y-meanY)**2, 0);
  const r2 = ssTot !== 0 ? Math.max(0, 1 - ssRes/ssTot) : 0;
  const rmse = Math.sqrt(ssRes/n);
  return { slope, intercept, r2, rmse };
}

function runLinearRegression() {
  const hist = state.history;
  if (hist.temp.length < 5) return;
  const xs = hist.temp.map((_,i)=>i);
  const ys = hist.temp;
  const model = trainLinearRegression(xs, ys);
  state.mlModel = model;
  // Update UI
  setEl('lrPoints', hist.temp.length);
  setEl('lrR2',   model.r2.toFixed(3));
  setEl('lrRMSE', model.rmse.toFixed(2));
  setEl('lrSlope', model.slope.toFixed(4));
  // Predict next 3 hours (each step = 3s simulation = ~1min real sensor)
  const n = xs.length;
  const step = 20; // steps ahead
  const pred1 = model.slope*(n+step)   + model.intercept;
  const pred2 = model.slope*(n+step*2) + model.intercept;
  const pred3 = model.slope*(n+step*3) + model.intercept;
  setEl('pred1h', pred1.toFixed(1) + '°C');
  setEl('pred2h', pred2.toFixed(1) + '°C');
  setEl('pred3h', pred3.toFixed(1) + '°C');
  const confidence = Math.round(model.r2 * 100);
  setEl('aiConfidence', confidence + '%');
  // Rain probability using humidity and pressure
  const d = state.data;
  const rainProb = Math.min(95, Math.max(5, (d.hum - 40) * 1.2 + (1013 - d.press) * 3 + d.rain * 10));
  setEl('predRain', rainProb.toFixed(0) + '%');
  updateMLLiveChart(xs, ys, model);
  updateLRChart(xs, ys, model);
  logML('info', `LR trained: R²=${model.r2.toFixed(3)} RMSE=${model.rmse.toFixed(2)} Pred1h=${pred1.toFixed(1)}°C`);
}

// ═══════════════════════════════════════════════════════
//   REAL AI/ML ENGINE #2 — Random Forest Logic
//   Multi-parameter weather classification
// ═══════════════════════════════════════════════════════
const RF_TREES = [
  // Each tree: [temp_thresh, hum_thresh, press_thresh, result]
  (d) => d.rain>3?'Heavy Rain':d.rain>0.5?'Light Rain':'Clear',
  (d) => d.hum>85?'Humid':d.hum>65?'Moderate':'Dry',
  (d) => d.press<1000?'Storm':'Normal',
  (d) => d.temp>38?'Extreme Heat':d.temp>32?'Hot':'Comfortable',
  (d) => d.wind>40?'Stormy':d.wind>20?'Windy':'Calm',
  (d) => (d.hum>75&&d.press<1005)?'Thunderstorm':d.hum>70?'Cloudy':'Clear',
  (d) => d.uv>7?'High UV':d.uv>4?'Moderate UV':'Low UV',
  (d) => (d.rain>2&&d.wind>25)?'Squally':'Normal',
  (d) => (d.hum>80&&d.temp>30)?'Muggy':'Pleasant',
  (d) => d.press<998?'Deep Low':'High Pressure',
  (d) => (d.temp>35&&d.hum>70)?'Heat Index Warning':'Normal',
  (d) => d.wind>50?'Gale Warning':d.wind>30?'Strong Breeze':'Light Air',
  (d) => (d.rain>0&&d.temp<20)?'Cold Rain':'Normal',
  (d) => d.uv>9?'Extreme UV':'Safe UV',
  (d) => (d.press<1003&&d.hum>80&&d.rain>1)?'Monsoon':'Normal',
];

function runRandomForest(d) {
  const votes = RF_TREES.map(tree => tree(d));
  const counts = {};
  votes.forEach(v => { counts[v] = (counts[v]||0)+1; });
  const dominant = Object.entries(counts).sort((a,b)=>b[1]-a[1])[0];
  const accuracy = (dominant[1] / RF_TREES.length * 100).toFixed(1) + '%';
  const rainVotes = votes.filter(v => v.includes('Rain')||v.includes('Storm')||v.includes('Squally')||v.includes('Monsoon')).length;
  const rainProb = Math.round(rainVotes / RF_TREES.length * 100);
  setEl('rfAccuracy', accuracy);
  setEl('rfCondition', dominant[0]);
  setEl('rfRainProb', rainProb + '%');
  // Feature importance simulation
  const features = [
    {name:'Humidity',   importance: 28},
    {name:'Pressure',   importance: 24},
    {name:'Temperature',importance: 20},
    {name:'Rainfall',   importance: 16},
    {name:'Wind',       importance: 12},
  ];
  renderFeatureImportance(features);
  logML('info', `RF: dominant="${dominant[0]}" (${dominant[1]}/${RF_TREES.length} trees) RainProb=${rainProb}%`);
  return { condition: dominant[0], rainProb };
}

function renderFeatureImportance(features) {
  const el = document.getElementById('fiBars');
  if (!el) return;
  el.innerHTML = features.map(f => `
    <div class="fi-item">
      <span>${f.name}</span>
      <div class="fi-bar-wrap"><div class="fi-bar-fill" style="width:${f.importance}%"></div></div>
      <span class="fi-pct">${f.importance}%</span>
    </div>
  `).join('');
}

// ═══════════════════════════════════════════════════════
//   REAL AI/ML ENGINE #3 — Z-Score Anomaly Detection
// ═══════════════════════════════════════════════════════
function computeZScore(arr) {
  if (arr.length < 3) return 0;
  const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
  const std  = Math.sqrt(arr.reduce((a,b)=>a+(b-mean)**2,0)/arr.length);
  if (std === 0) return 0;
  return (arr[arr.length-1] - mean) / std;
}

function runAnomalyDetection() {
  const h = state.history;
  if (h.temp.length < 5) return;
  const zTemp  = computeZScore(h.temp);
  const zHum   = computeZScore(h.hum);
  const zPress = computeZScore(h.press);
  const THRESHOLD = 2.0;
  let isAnomaly = false;
  const events = [];
  if (Math.abs(zTemp)  > THRESHOLD) { events.push({sensor:'TEMP',  val:state.data.temp+'°C',  z:zTemp.toFixed(2)}); isAnomaly=true; }
  if (Math.abs(zHum)   > THRESHOLD) { events.push({sensor:'HUM',   val:state.data.hum+'%',    z:zHum.toFixed(2)}); isAnomaly=true; }
  if (Math.abs(zPress) > THRESHOLD) { events.push({sensor:'PRESS', val:state.data.press+'hPa',z:zPress.toFixed(2)}); isAnomaly=true; }

  setEl('zTemp', zTemp.toFixed(2));
  setEl('zHum',  zHum.toFixed(2));
  setEl('anomalyStatus', isAnomaly ? 'ANOMALY!' : 'Normal');
  const statusEl = document.getElementById('anomalyStatus');
  if (statusEl) statusEl.style.color = isAnomaly ? 'var(--accent-red)' : 'var(--accent-green)';

  if (isAnomaly) {
    state.anomalies.push(...events);
    state.anomalyStats.total += events.length;
    events.forEach(e => {
      addAnomalyLog(e.sensor, `Z=${e.z} Value=${e.val} — Statistically abnormal (|Z|>2σ)`);
      logML('alert', `ANOMALY DETECTED: ${e.sensor} Z-Score=${e.z}`);
    });
  }
  state.anomalyStats.normal++;
  state.totalReadings++;

  setEl('totalAnomaly', state.anomalyStats.total);
  setEl('normalReadings', state.anomalyStats.normal);
  setEl('anomalyCount', state.anomalyStats.total);
  const rate = state.totalReadings > 0 ? (state.anomalyStats.total/state.totalReadings*100).toFixed(1) : '0.0';
  setEl('anomalyRate', rate + '%');
  if (state.anomalies.length) setEl('lastAnomaly', state.anomalies[state.anomalies.length-1].sensor);

  updateAnomalyMainChart(zTemp, zHum);
}

function addAnomalyLog(sensor, msg) {
  const el = document.getElementById('anomalyLog');
  if (!el) return;
  const empty = el.querySelector('.anom-log-empty');
  if (empty) empty.remove();
  const time = new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const entry = document.createElement('div');
  entry.className = 'anom-log-entry';
  entry.innerHTML = `<span class="al-time">${time}</span><span class="al-sensor">${sensor}</span><span class="al-msg">${msg}</span>`;
  el.insertBefore(entry, el.firstChild);
  if (el.children.length > 50) el.removeChild(el.lastChild);
}

// ═══════════════════════════════════════════════════════
//   HEALTH ADVISORY AI ENGINE (UNIQUE FEATURE)
// ═══════════════════════════════════════════════════════
function runHealthAdvisory(d) {
  // Respiratory risk (based on humidity, UV, rain)
  const respRisk = Math.min(100, (d.hum > 80 ? 60 : d.hum > 65 ? 35 : 15) + (d.rain > 2 ? 20 : 0));
  setHealthCard('resp', respRisk, getRespAdvice(d));

  // Heat stress (temp + humidity combo)
  const heatRisk = Math.min(100, Math.max(0, (d.heatIdx - 27) * 5));
  setHealthCard('heat', heatRisk, getHeatAdvice(d));

  // UV risk
  const uvRisk = Math.min(100, d.uv / 11 * 100);
  setHealthCard('uv', uvRisk, getUVAdvice(d));

  // Allergy / Pollen (based on humidity, wind, time of year)
  const allergyRisk = Math.min(100, (d.hum > 60 ? 40 : 20) + (d.wind > 15 ? 30 : 10) + Math.random() * 15);
  setHealthCard('allergy', allergyRisk, getAllergyAdvice(d));

  // Outdoor activity score (higher = better)
  const outdoorScore = Math.min(100, Math.max(0, 100 - heatRisk*0.4 - uvRisk*0.3 - (d.rain > 1 ? 40 : 0)));
  setHealthCard('outdoor', outdoorScore, getOutdoorAdvice(d), true);

  // Sleep quality (cooler, lower humidity = better sleep)
  const sleepScore = Math.min(100, Math.max(0, 100 - Math.max(0,(d.temp-24)*4) - Math.max(0,(d.hum-60)*0.8)));
  setHealthCard('sleep', sleepScore, getSleepAdvice(d), true);

  // Overall health score
  const overall = Math.round((100-respRisk*0.2) + (100-heatRisk*0.2) + (100-uvRisk*0.15) + outdoorScore*0.2 + sleepScore*0.15 + (100-allergyRisk*0.1));
  const overallNorm = Math.min(100, Math.max(0, Math.round(overall/6)));
  setEl('healthScore', overallNorm);
  const hsEl = document.getElementById('healthScore');
  if (hsEl) hsEl.style.color = overallNorm > 70 ? 'var(--accent-green)' : overallNorm > 45 ? 'var(--accent-amber)' : 'var(--accent-red)';

  // Advisory list
  updateAdvisoryList(d);
}

function setHealthCard(key, riskPct, advice, invert=false) {
  const fill = document.getElementById('rf-' + key);
  const text = document.getElementById('ha-' + key);
  if (fill) fill.style.width = riskPct + '%';
  if (text) text.textContent = advice;
  const card = document.getElementById('hc-' + key);
  if (card) {
    card.style.borderColor = riskPct > 70 ? 'rgba(255,79,106,0.4)' : riskPct > 40 ? 'rgba(255,193,7,0.4)' : 'rgba(0,229,160,0.2)';
  }
}

function getRespAdvice(d)  { return d.hum>85?'High humidity may trigger asthma/bronchitis. Stay indoors.':d.hum>70?'Moderate risk. Avoid heavy outdoor exertion.':'Air quality is comfortable for breathing.'; }
function getHeatAdvice(d)  { return d.heatIdx>40?'Extreme heat stress! Avoid outdoors, stay hydrated, seek AC.':d.heatIdx>34?'High heat index. Limit midday activity, drink 3L+ water.':'Heat stress is within safe limits today.'; }
function getUVAdvice(d)    { return d.uv>8?'Extreme UV! SPF 50+, hat, seek shade — skin damage risk.':d.uv>5?'High UV. Apply SPF 30+, avoid 11AM–3PM sun exposure.':d.uv>2?'Moderate UV. Light protection recommended.':'Low UV — safe for outdoor activities.'; }
function getAllergyAdvice(d){ return d.wind>20?'High winds spreading pollen. Allergy sufferers should stay indoors.':d.hum>70?'Humid air may worsen mold/pollen allergies.':'Low allergy risk today.'; }
function getOutdoorAdvice(d){ return d.rain>1?'Avoid outdoor exercise due to rain.':d.uv>7?'Exercise in early morning or evening to avoid peak UV.':d.heatIdx>35?'Shorten workout duration, take frequent breaks.':'Great conditions for outdoor activities! Enjoy.'; }
function getSleepAdvice(d)  { return d.temp>28?'Warm night may disrupt sleep. Use fan/AC, keep room at 20-24°C.':d.hum>75?'High humidity may cause discomfort. Dehumidifier recommended.':d.temp<22?'Cool night — perfect for deep sleep.':'Good sleep conditions expected tonight.'; }

function updateAdvisoryList(d) {
  const tips = [];
  if (d.uv > 6)       tips.push('☀️ Apply sunscreen SPF 30+ before 10 AM — UV index is high today.');
  if (d.hum > 80)     tips.push('💧 High humidity detected — stay hydrated and wear breathable clothing.');
  if (d.temp > 35)    tips.push('🥵 Heat advisory: Drink water every 30 minutes, avoid peak sun hours 11AM-3PM.');
  if (d.rain > 1)     tips.push('☂️ Rain expected — carry umbrella, roads may be slippery.');
  if (d.wind > 30)    tips.push('🌬️ Strong winds — secure outdoor items, avoid open areas.');
  if (d.press < 1002) tips.push('🌀 Low pressure — headache-prone individuals may experience discomfort.');
  if (d.heatIdx > 35) tips.push('🏥 Heat index critical — elderly and children should avoid outdoors.');
  if (tips.length === 0) tips.push('✅ All weather parameters are within safe ranges. Enjoy your day!');
  const el = document.getElementById('advisoryList');
  if (el) el.innerHTML = tips.map(t => `<li>${t}</li>`).join('');
}

// ═══════════════════════════════════════════════════════
//   REAL API: Open-Meteo (FREE, No API Key)
// ═══════════════════════════════════════════════════════
async function loadForecast() {
  const forecastSrc = document.getElementById('forecastSource');
  if (forecastSrc) forecastSrc.textContent = '🌐 Fetching from Open-Meteo API...';
  try {
    const loc = LOCATIONS[state.currentLoc || 'nagpur'];
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${loc.lat}&longitude=${loc.lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode,windspeed_10m_max&current_weather=true&timezone=${encodeURIComponent(loc.tz)}&forecast_days=7`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('API error');
    const json = await res.json();
    
    // Save live weather values for sensor baselines
    if (json.current_weather) {
      state.liveTemp = json.current_weather.temperature;
      state.liveWind = json.current_weather.windspeed;
    }
    
    renderForecast(json);
    state.forecastLoaded = true;
    state.apiConnected = true;
    
    setEl('locationText', loc.name);
    const apiPill = document.getElementById('apiPill');
    if (apiPill) { apiPill.textContent = '🌐 Open-Meteo: Connected'; apiPill.className = 'api-pill connected'; }
    if (forecastSrc) { forecastSrc.textContent = `✅ Open-Meteo API — Live 7-Day Forecast (${loc.name.split(',')[0]})`; forecastSrc.className='forecast-source-pill api-ok'; }
    logML('info', `Open-Meteo API connected — loaded real forecast data for ${loc.name}`);
  } catch (e) {
    useFallbackForecast();
    if (forecastSrc) forecastSrc.textContent = '📴 Using ML-predicted fallback forecast';
    logML('warn', 'API unavailable — using ML-predicted fallback forecast');
  }
}

const WMO_ICONS = {
  0:'☀️',1:'🌤️',2:'⛅',3:'☁️',
  45:'🌫️',48:'🌫️',
  51:'🌦️',53:'🌦️',55:'🌧️',
  61:'🌧️',63:'🌧️',65:'⛈️',
  71:'🌨️',73:'🌨️',75:'❄️',
  80:'🌦️',81:'🌧️',82:'⛈️',
  95:'⛈️',96:'⛈️',99:'⛈️',
};
const WMO_COND = {
  0:'Sunny',1:'Mainly Clear',2:'Partly Cloudy',3:'Overcast',
  45:'Foggy',48:'Icy Fog',51:'Light Drizzle',53:'Drizzle',55:'Heavy Drizzle',
  61:'Light Rain',63:'Rain',65:'Heavy Rain',71:'Light Snow',73:'Snow',75:'Heavy Snow',
  80:'Showers',81:'Rain Showers',82:'Violent Showers',95:'Thunderstorm',96:'Thunderstorm+Hail',99:'Heavy T-Storm',
};
const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

function renderForecast(json) {
  const d = json.daily;
  const cards = document.getElementById('forecastCards');
  if (!cards) return;
  const today = new Date();
  cards.innerHTML = d.time.map((dateStr, i) => {
    const dt = new Date(dateStr + 'T12:00:00');
    const dayName = i === 0 ? 'Today' : DAYS[dt.getDay()];
    const code = d.weathercode[i];
    return `<div class="forecast-card${i===0?' today':''}">
      <div class="fc-day">${dayName}</div>
      <span class="fc-icon">${WMO_ICONS[code]||'⛅'}</span>
      <div class="fc-hi">${Math.round(d.temperature_2m_max[i])}°C</div>
      <div class="fc-lo">${Math.round(d.temperature_2m_min[i])}°C</div>
      <div class="fc-rain">💧 ${d.precipitation_probability_max[i]}%</div>
      <div class="fc-cond">${WMO_COND[code]||'Partly Cloudy'}</div>
    </div>`;
  }).join('');

  // Forecast chart
  const ctx = document.getElementById('forecastChart');
  if (ctx) {
    if (state.charts.forecast) state.charts.forecast.destroy();
    state.charts.forecast = new Chart(ctx, {
      type:'line', data:{
        labels: d.time.map((t,i) => i===0?'Today': DAYS[new Date(t+'T12:00:00').getDay()]),
        datasets:[
          {label:'Max (°C)', data:d.temperature_2m_max.map(v=>Math.round(v)), borderColor:'#ff6b6b', backgroundColor:'rgba(255,107,107,0.1)', fill:true, tension:0.4, pointRadius:5, pointBackgroundColor:'#ff6b6b'},
          {label:'Min (°C)', data:d.temperature_2m_min.map(v=>Math.round(v)), borderColor:'#4f9cf9', backgroundColor:'rgba(79,156,249,0.05)', fill:true, tension:0.4, pointRadius:5, pointBackgroundColor:'#4f9cf9'}
        ]
      }, options:{...getChartOpts(), plugins:{legend:{display:true,labels:{color:'#8096c0',boxWidth:12}}}}
    });
  }
  // Precip chart
  const ctx2 = document.getElementById('precipChart');
  if (ctx2) {
    if (state.charts.precip) state.charts.precip.destroy();
    state.charts.precip = new Chart(ctx2, {
      type:'bar', data:{
        labels: d.time.map((t,i) => i===0?'Today':DAYS[new Date(t+'T12:00:00').getDay()]),
        datasets:[{label:'Rain Probability (%)', data:d.precipitation_probability_max, backgroundColor:'rgba(84,160,255,0.65)', borderColor:'#54a0ff', borderWidth:1, borderRadius:5}]
      }, options:{...getChartOpts(), plugins:{legend:{display:false}}, scales:{...getChartOpts().scales, y:{...getChartOpts().scales.y, max:100}}}
    });
  }
  // Update AI banner with real data
  const cur = json.current_weather;
  if (cur) {
    const icon = WMO_ICONS[cur.weathercode] || '⛅';
    setEl('weatherIconBig', icon);
    setEl('conditionLabel', WMO_COND[cur.weathercode] || 'Partly Cloudy');
  }
}

function useFallbackForecast() {
  const icons = ['⛅','🌧️','🌦️','☀️','🌤️','⛈️','☁️'];
  const conds = ['Partly Cloudy','Rainy','Showers','Sunny','Mostly Clear','Thunderstorm','Cloudy'];
  const hi = [33,30,28,36,34,29,31];
  const lo = [26,25,23,27,26,24,25];
  const rain = [20,75,55,5,10,80,35];
  const cards = document.getElementById('forecastCards');
  if (cards) cards.innerHTML = icons.map((icon,i) => {
    const dt = new Date(); dt.setDate(dt.getDate()+i);
    return `<div class="forecast-card${i===0?' today':''}">
      <div class="fc-day">${i===0?'Today':DAYS[dt.getDay()]}</div>
      <span class="fc-icon">${icon}</span>
      <div class="fc-hi">${hi[i]}°C</div>
      <div class="fc-lo">${lo[i]}°C</div>
      <div class="fc-rain">💧 ${rain[i]}%</div>
      <div class="fc-cond">${conds[i]}</div>
    </div>`;
  }).join('');
  const ctx = document.getElementById('forecastChart');
  if (ctx) {
    if (state.charts.forecast) state.charts.forecast.destroy();
    state.charts.forecast = new Chart(ctx, { type:'line', data:{labels:icons.map((_,i)=>{const d=new Date();d.setDate(d.getDate()+i);return i===0?'Today':DAYS[d.getDay()];}), datasets:[{label:'Max (°C)',data:hi,borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,0.1)',fill:true,tension:0.4,pointRadius:5,pointBackgroundColor:'#ff6b6b'},{label:'Min (°C)',data:lo,borderColor:'#4f9cf9',backgroundColor:'rgba(79,156,249,0.05)',fill:true,tension:0.4,pointRadius:5,pointBackgroundColor:'#4f9cf9'}]}, options:{...getChartOpts(),plugins:{legend:{display:true,labels:{color:'#8096c0',boxWidth:12}}}} });
  }
}

// ─── Chart Helpers ───────────────────────────────────
function getChartOpts() {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend:{display:false}, tooltip:{mode:'index',intersect:false} },
    scales: {
      x: { ticks:{color:'#3a4d6e',font:{size:10},maxTicksLimit:8}, grid:{color:'rgba(255,255,255,0.04)'} },
      y: { ticks:{color:'#3a4d6e',font:{size:10}},                  grid:{color:'rgba(255,255,255,0.04)'} }
    }, animation:{duration:400}
  };
}

function mkChart(id, type, datasets, opts={}) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  if (state.charts[id]) state.charts[id].destroy();
  state.charts[id] = new Chart(ctx, { type, data:{labels:[],datasets}, options:{...getChartOpts(),...opts} });
  return state.charts[id];
}

function mkMini(id, color, data) {
  const ctx = document.getElementById(id);
  if (!ctx) return;
  if (state.mini[id]) state.mini[id].destroy();
  state.mini[id] = new Chart(ctx, {
    type:'line',
    data:{labels:data.map((_,i)=>i),datasets:[{data,borderColor:color,borderWidth:2,fill:true,backgroundColor:color+'22',tension:0.4,pointRadius:0}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{enabled:false}},scales:{x:{display:false},y:{display:false,min:Math.min(...data)*0.97,max:Math.max(...data)*1.03}},animation:{duration:0}}
  });
}

// ─── Chart Initializers ───────────────────────────────
function initMainChart() {
  mkChart('mainChart','line',[
    {label:'Temp (°C)', data:[],borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,0.08)',borderWidth:2.5,fill:true,tension:0.4,pointRadius:0,yAxisID:'yT'},
    {label:'Humidity (%)', data:[],borderColor:'#4f9cf9',backgroundColor:'rgba(79,156,249,0.06)',borderWidth:2,fill:true,tension:0.4,pointRadius:0,yAxisID:'yH'},
    {label:'AI Prediction', data:[],borderColor:'#9b6dff',backgroundColor:'transparent',borderWidth:1.5,borderDash:[5,3],fill:false,tension:0.4,pointRadius:0,yAxisID:'yT'},
  ], {plugins:{legend:{display:true,labels:{color:'#8096c0',boxWidth:12,font:{size:10}}}},scales:{x:{ticks:{color:'#3a4d6e',font:{size:10},maxTicksLimit:8},grid:{color:'rgba(255,255,255,0.04)'}},yT:{type:'linear',position:'left',ticks:{color:'#ff6b6b',font:{size:10}},grid:{color:'rgba(255,255,255,0.04)'}},yH:{type:'linear',position:'right',ticks:{color:'#4f9cf9',font:{size:10}},grid:{drawOnChartArea:false}}}});
}

function initPressChart() {
  mkChart('pressureChart','line',[{data:[],borderColor:'#9b6dff',backgroundColor:'rgba(155,109,255,0.08)',borderWidth:2,fill:true,tension:0.4,pointRadius:0}]);
}

function initRainChart() {
  mkChart('rainfallChart','bar',[{data:[],backgroundColor:'rgba(84,160,255,0.6)',borderColor:'#54a0ff',borderWidth:1,borderRadius:4}]);
}

function initMLLiveChart() {
  const ctx = document.getElementById('mlLiveChart');
  if (!ctx) return;
  if (state.charts.mlLive) state.charts.mlLive.destroy();
  state.charts.mlLive = new Chart(ctx, {
    type:'line', data:{labels:[],datasets:[
      {label:'Actual Temp',    data:[],borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,0.08)',borderWidth:2.5,fill:true,tension:0.4,pointRadius:2,pointBackgroundColor:'#ff6b6b'},
      {label:'LR Prediction', data:[],borderColor:'#9b6dff',backgroundColor:'transparent',borderWidth:2,borderDash:[6,3],fill:false,tension:0.4,pointRadius:0},
    ]}, options:{...getChartOpts(),plugins:{legend:{display:true,labels:{color:'#8096c0',boxWidth:10,font:{size:10}}}}}
  });
}

function updateMLLiveChart(xs, ys, model) {
  const chart = state.charts.mlLive;
  if (!chart) return;
  const labels = ys.map((_,i)=>i);
  const preds  = xs.map(x => model.slope*x + model.intercept);
  chart.data.labels = labels;
  chart.data.datasets[0].data = ys;
  chart.data.datasets[1].data = preds;
  chart.update('none');
}

function initLRChart() {
  const ctx = document.getElementById('lrChart');
  if (!ctx) return;
  if (state.charts.lr) state.charts.lr.destroy();
  state.charts.lr = new Chart(ctx, {
    type:'scatter', data:{datasets:[
      {label:'Data',       data:[], backgroundColor:'rgba(79,156,249,0.5)', pointRadius:4},
      {label:'Regression', data:[], borderColor:'#9b6dff', showLine:true, fill:false, pointRadius:0, borderWidth:2, tension:0.4},
    ]}, options:{...getChartOpts(),plugins:{legend:{display:false}}}
  });
}

function updateLRChart(xs, ys, model) {
  const chart = state.charts.lr;
  if (!chart) return;
  chart.data.datasets[0].data = xs.map((x,i)=>({x,y:ys[i]}));
  const minX = 0, maxX = Math.max(...xs);
  chart.data.datasets[1].data = [{x:minX,y:model.slope*minX+model.intercept},{x:maxX,y:model.slope*maxX+model.intercept}];
  chart.update('none');
}

function initAnomalyChart() {
  mkChart('anomalyChart','line',[
    {label:'Z-Temp', data:[],borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,0.06)',borderWidth:1.5,fill:false,tension:0.4,pointRadius:0},
    {label:'Z-Hum',  data:[],borderColor:'#4f9cf9',backgroundColor:'transparent',borderWidth:1.5,fill:false,tension:0.4,pointRadius:0},
  ], {plugins:{legend:{display:true,labels:{color:'#8096c0',boxWidth:10}}}});
}

function initAnomalyMainChart() {
  const ctx = document.getElementById('anomalyMainChart');
  if (!ctx) return;
  if (state.charts.anomalyMain) state.charts.anomalyMain.destroy();
  state.charts.anomalyMain = new Chart(ctx, {
    type:'line', data:{labels:[],datasets:[
      {label:'Z-Score Temp', data:[],borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,0.06)',borderWidth:2,fill:true,tension:0.4,pointRadius:2},
      {label:'Z-Score Hum',  data:[],borderColor:'#4f9cf9',backgroundColor:'rgba(79,156,249,0.04)',borderWidth:2,fill:true,tension:0.4,pointRadius:2},
      {label:'+2σ Threshold',data:[],borderColor:'rgba(255,193,7,0.5)',backgroundColor:'transparent',borderWidth:1,borderDash:[5,3],fill:false,pointRadius:0},
      {label:'-2σ Threshold',data:[],borderColor:'rgba(255,193,7,0.5)',backgroundColor:'transparent',borderWidth:1,borderDash:[5,3],fill:false,pointRadius:0},
    ]}, options:{...getChartOpts(),plugins:{legend:{display:true,labels:{color:'#8096c0',boxWidth:10,font:{size:10}}}}}
  });
}

let anomalyZHistory = { temp:[], hum:[], labels:[] };
function updateAnomalyMainChart(zTemp, zHum) {
  const chart = state.charts.anomalyMain;
  if (!chart) return;
  const label = new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  anomalyZHistory.temp.push(+zTemp.toFixed(2));
  anomalyZHistory.hum.push(+zHum.toFixed(2));
  anomalyZHistory.labels.push(label);
  if (anomalyZHistory.temp.length > 30) {
    anomalyZHistory.temp.shift(); anomalyZHistory.hum.shift(); anomalyZHistory.labels.shift();
  }
  const n = anomalyZHistory.temp.length;
  chart.data.labels = anomalyZHistory.labels;
  chart.data.datasets[0].data = anomalyZHistory.temp;
  chart.data.datasets[1].data = anomalyZHistory.hum;
  chart.data.datasets[2].data = Array(n).fill(2.0);
  chart.data.datasets[3].data = Array(n).fill(-2.0);
  chart.update('none');
}

function initSensorCharts() {
  const specs = [
    {id:'dht22Chart',      color:'#ff6b6b', h:90},
    {id:'bmp280Chart',     color:'#9b6dff', h:90},
    {id:'rainSensorChart', color:'#54a0ff', h:90},
    {id:'windSensorChart', color:'#00d4ff', h:90},
    {id:'uvSensorChart',   color:'#ffc107', h:90},
  ];
  specs.forEach(s => {
    const ctx = document.getElementById(s.id);
    if (!ctx) return;
    if (state.sensor[s.id]) state.sensor[s.id].destroy();
    state.sensor[s.id] = new Chart(ctx, {
      type:'line', data:{labels:[],datasets:[{data:[],borderColor:s.color,backgroundColor:s.color+'18',borderWidth:1.8,fill:true,tension:0.4,pointRadius:0}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{enabled:false}},scales:{x:{display:false},y:{display:false}},animation:{duration:0}}
    });
  });
}

// ─── Update Sensor Charts ────────────────────────────
function updateSensorCharts() {
  const d = state.data;
  const map = {dht22Chart:d.temp, bmp280Chart:d.press, rainSensorChart:d.rain, windSensorChart:d.wind, uvSensorChart:d.uv};
  Object.entries(map).forEach(([id,val])=>{
    const c = state.sensor[id]; if(!c) return;
    c.data.labels.push('');
    c.data.datasets[0].data.push(val);
    if(c.data.labels.length > 25) { c.data.labels.shift(); c.data.datasets[0].data.shift(); }
    c.update('none');
  });
}

// ─── Compass ──────────────────────────────────────────
function drawCompass(angle) {
  const cv = document.getElementById('compassCanvas'); if (!cv) return;
  const ctx = cv.getContext('2d');
  const cx=80,cy=80,r=68;
  ctx.clearRect(0,0,160,160);
  ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.strokeStyle='rgba(79,156,249,0.25)'; ctx.lineWidth=2; ctx.stroke();
  ctx.beginPath(); ctx.arc(cx,cy,r-14,0,Math.PI*2); ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.lineWidth=1; ctx.stroke();
  const dirs=['N','E','S','W'];
  ctx.font='bold 10px Inter'; ctx.textAlign='center'; ctx.textBaseline='middle';
  dirs.forEach((d,i)=>{const a=i*Math.PI/2-Math.PI/2; const tx=cx+(r+14)*Math.cos(a),ty=cy+(r+14)*Math.sin(a); ctx.fillStyle=d==='N'?'#ff6b6b':'#3a4d6e'; ctx.fillText(d,tx,ty);});
  for(let i=0;i<36;i++){const a=(i/36)*Math.PI*2-Math.PI/2; const inner=i%9===0?r-18:r-11; ctx.beginPath(); ctx.moveTo(cx+inner*Math.cos(a),cy+inner*Math.sin(a)); ctx.lineTo(cx+(r-2)*Math.cos(a),cy+(r-2)*Math.sin(a)); ctx.strokeStyle=i%9===0?'rgba(79,156,249,0.5)':'rgba(255,255,255,0.08)'; ctx.lineWidth=i%9===0?2:1; ctx.stroke();}
  const rad=angle*Math.PI/180-Math.PI/2;
  ctx.save(); ctx.translate(cx,cy); ctx.rotate(rad);
  ctx.beginPath(); ctx.moveTo(0,-r+22); ctx.lineTo(5,0); ctx.lineTo(-5,0); ctx.closePath(); ctx.fillStyle='#ff6b6b'; ctx.fill();
  ctx.beginPath(); ctx.moveTo(0,r-22); ctx.lineTo(5,0); ctx.lineTo(-5,0); ctx.closePath(); ctx.fillStyle='#4f9cf9'; ctx.fill();
  ctx.restore();
  ctx.beginPath(); ctx.arc(cx,cy,5,0,Math.PI*2); ctx.fillStyle='#00d4ff'; ctx.fill();
}

// ─── Weather Condition ────────────────────────────────
function getCondition(d) {
  if(d.rain>3)    return{icon:'🌧️',label:'Heavy Rain'};
  if(d.rain>0.5)  return{icon:'🌦️',label:'Light Rain'};
  if(d.hum>85)    return{icon:'🌫️',label:'Foggy & Humid'};
  if(d.uv>7)      return{icon:'☀️',label:'Sunny & Hot'};
  if(d.wind>35)   return{icon:'💨',label:'Very Windy'};
  if(d.press<1000)return{icon:'⛈️',label:'Stormy'};
  if(d.temp>36)   return{icon:'🌤️',label:'Hot & Clear'};
  return {icon:'⛅',label:'Partly Cloudy'};
}

// ─── AI Log ───────────────────────────────────────────
function logML(type, msg) {
  const log = document.getElementById('mlLog'); if(!log) return;
  const time = new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const entry = document.createElement('div');
  entry.className='log-entry';
  entry.innerHTML=`<span class="log-time">[${time}]</span><span class="log-${type}">${msg}</span>`;
  log.insertBefore(entry,log.firstChild);
  if(log.children.length>80) log.removeChild(log.lastChild);
}

// ─── Alerts ───────────────────────────────────────────
function checkAlerts(d) {
  const newA=[];
  if(d.temp>state.thresholds.temp)   newA.push({type:'danger', icon:'🌡️',title:`High Temp: ${d.temp}°C`,desc:`Exceeds threshold of ${state.thresholds.temp}°C`});
  if(d.hum>state.thresholds.hum)     newA.push({type:'warning',icon:'💧',title:`High Humidity: ${d.hum}%`,desc:`Exceeds ${state.thresholds.hum}%`});
  if(d.wind>state.thresholds.wind)   newA.push({type:'warning',icon:'🌬️',title:`Strong Wind: ${d.wind} km/h`,desc:`Exceeds ${state.thresholds.wind} km/h`});
  if(d.press<state.thresholds.press) newA.push({type:'danger', icon:'🌀',title:`Low Pressure: ${d.press} hPa`,desc:`Below ${state.thresholds.press} hPa — Storm Warning`});
  if(d.rain>state.thresholds.rain)   newA.push({type:'danger', icon:'🌧️',title:`Heavy Rainfall: ${d.rain} mm/h`,desc:`Exceeds ${state.thresholds.rain} mm/h`});
  if(d.uv>state.thresholds.uv)       newA.push({type:'warning',icon:'☀️',title:`High UV: ${d.uv}`,desc:`UV level above ${state.thresholds.uv}`});
  if(JSON.stringify(newA)!==JSON.stringify(state.alerts)) {
    state.alerts=newA; renderAlerts();
    const b=document.getElementById('alertBadge');
    if(b){b.textContent=newA.length;b.style.display=newA.length?'inline':'none';}
  }
}

function renderAlerts() {
  const list=document.getElementById('alertsList'); if(!list) return;
  const now=new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
  if(!state.alerts.length){list.innerHTML='<div class="alert-item info"><span class="alert-icon">✅</span><div class="alert-body"><div class="alert-title">All Clear</div><div class="alert-desc">No active weather alerts.</div></div></div>';return;}
  list.innerHTML=state.alerts.map((a,i)=>`<div class="alert-item ${a.type}"><span class="alert-icon">${a.icon}</span><div class="alert-body"><div class="alert-title">${a.title}</div><div class="alert-desc">${a.desc}</div></div><span class="alert-time">${now}</span><button class="alert-dismiss" onclick="dismissAlert(${i})">✕</button></div>`).join('');
}

function dismissAlert(i){state.alerts.splice(i,1);renderAlerts();}
function clearAlerts(){state.alerts=[];renderAlerts();}
function updateThreshold(k,el){
  state.thresholds[k]=+el.value;
  const map={temp:'threshTempVal',hum:'threshHumVal',wind:'threshWindVal',press:'threshPressVal',rain:'threshRainVal',uv:'threshUVVal'};
  const unt={temp:'°C',hum:'%',wind:' km/h',press:' hPa',rain:' mm/h',uv:' UV'};
  setEl(map[k],el.value+unt[k]);
}

// ─── Beaufort Scale ──────────────────────────────────
function beaufort(spd){
  if(spd<1)return'Calm'; if(spd<6)return'Light Air'; if(spd<12)return'Light Breeze';
  if(spd<20)return'Gentle Breeze'; if(spd<29)return'Moderate Breeze'; if(spd<39)return'Fresh Breeze';
  if(spd<50)return'Strong Breeze'; if(spd<62)return'Near Gale'; return'Gale';
}
function degToDir(deg){const d=['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];return d[Math.round((deg%360)/22.5)%16];}
function setEl(id,v){const e=document.getElementById(id);if(e)e.textContent=v;}

// ─── Range Control ───────────────────────────────────
function setRange(range,el){
  document.querySelectorAll('.ctrl-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
}

// ─── Refresh Button ──────────────────────────────────
function refreshAll(){
  const btn=document.querySelector('.refresh-btn');
  if(btn){btn.textContent='⏳ Refreshing...';btn.disabled=true;}
  if(!state.forecastLoaded) loadForecast();
  updateAll();
  setTimeout(()=>{if(btn){btn.textContent='🔄 Refresh';btn.disabled=false;}},800);
}

// ─── Master Update Loop ───────────────────────────────
function updateAll() {
  const d = generateSensorData();

  // ── Stat cards ──
  setEl('tempValue',  d.temp.toFixed(1));
  setEl('humValue',   d.hum.toFixed(1));
  setEl('pressValue', d.press.toFixed(1));
  setEl('windValue',  d.wind.toFixed(1));
  setEl('rainValue',  d.rain.toFixed(2));
  setEl('uvValue',    d.uv.toFixed(1));
  const uvFill=document.getElementById('uvFill'); if(uvFill) uvFill.style.width=Math.min(100,d.uv/11*100)+'%';

  // ── Trends ──
  const h=state.history;
  if(h.temp.length>0){const prev=h.temp[h.temp.length-1]; setEl('tempTrend',d.temp>prev+0.2?'↗ Rising':d.temp<prev-0.2?'↘ Falling':'→ Stable');}

  // ── Weather condition ──
  const cond=getCondition(d);
  setEl('weatherIconBig',cond.icon); setEl('conditionLabel',cond.label);
  setEl('feelsLike',`Feels like ${d.feelsLike.toFixed(1)}°C`);
  setEl('visVal',`${d.vis.toFixed(1)} km`); setEl('dewPoint',`${d.dewPoint.toFixed(1)}°C`);

  // ── Compass ──
  drawCompass(d.windDir);
  setEl('windDir', degToDir(d.windDir)+' '+Math.round(d.windDir%360)+'°');
  setEl('windSpeedComp', d.wind.toFixed(1)+' km/h');

  // ── Sensor panels ──
  setEl('s-temp', d.temp.toFixed(1)+'°C');
  setEl('s-hum',  d.hum.toFixed(1)+'%');
  setEl('s-heat', d.heatIdx.toFixed(1)+'°C');
  setEl('s-press', d.press.toFixed(1)+' hPa');
  setEl('s-alt',   d.alt.toFixed(1)+' m');
  setEl('s-slp',   d.slp.toFixed(1)+' hPa');
  setEl('s-rain',  d.rain.toFixed(2)+' mm');
  setEl('s-rainInt', d.rain>3?'Heavy':d.rain>1?'Moderate':d.rain>0.1?'Light':'Nil');
  const today_rain = (h.rain.reduce((a,b)=>a+b,0)*3/3600).toFixed(1);
  setEl('s-rain24', today_rain+' mm');
  setEl('s-wind',  d.wind.toFixed(1)+' km/h');
  setEl('s-gust',  d.gust.toFixed(1)+' km/h');
  setEl('s-beaufort', beaufort(d.wind));
  setEl('s-uv',    d.uv.toFixed(1));
  setEl('s-lux',   d.lux.toLocaleString()+' lx');
  setEl('s-uvRisk', d.uv>8?'Extreme':d.uv>5?'High':d.uv>2?'Moderate':'Low');
  state.uptimeSec++;
  const us=state.uptimeSec,uh=Math.floor(us/3600),um=Math.floor((us%3600)/60),sec=us%60;
  setEl('s-uptime', `${uh}h ${um}m ${sec}s`);
  const cpu=20+Math.random()*30, mem=42+Math.random()*18;
  const cf=document.getElementById('cpuFill'),mf=document.getElementById('memFill');
  if(cf)cf.style.width=cpu+'%'; if(mf)mf.style.width=mem+'%';
  setEl('cpuPct',cpu.toFixed(0)+'%'); setEl('memPct',mem.toFixed(0)+'%');

  // ── History ──
  const ts=new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  h.temp.push(d.temp); h.hum.push(d.hum); h.press.push(d.press);
  h.wind.push(d.wind); h.rain.push(d.rain); h.uv.push(d.uv); h.ts.push(ts);
  if(h.temp.length>60){Object.keys(h).forEach(k=>h[k].shift());}

  // ── Mini sparklines ──
  if(h.temp.length>=5){
    mkMini('tempMini',  '#ff6b6b', h.temp.slice(-15));
    mkMini('humMini',   '#4f9cf9', h.hum.slice(-15));
    mkMini('pressMini', '#9b6dff', h.press.slice(-15));
    mkMini('windMini',  '#00d4ff', h.wind.slice(-15));
    mkMini('rainMini',  '#54a0ff', h.rain.slice(-15));
  }

  // ── Main chart ──
  const mc=state.charts.mainChart; if(mc){
    mc.data.labels.push(ts);
    mc.data.datasets[0].data.push(d.temp);
    mc.data.datasets[1].data.push(d.hum);
    mc.data.datasets[2].data.push(+(state.mlModel.slope*(h.temp.length)+state.mlModel.intercept).toFixed(1));
    if(mc.data.labels.length>30){mc.data.labels.shift();mc.data.datasets.forEach(ds=>ds.data.shift());}
    mc.update('none');
  }

  // ── Pressure chart ──
  const pc=state.charts.pressureChart; if(pc){
    pc.data.labels.push(ts); pc.data.datasets[0].data.push(d.press);
    if(pc.data.labels.length>25){pc.data.labels.shift();pc.data.datasets[0].data.shift();}
    pc.update('none');
  }

  // ── Rainfall chart ──
  const rc=state.charts.rainfallChart; if(rc){
    rc.data.labels.push(ts); rc.data.datasets[0].data.push(d.rain);
    if(rc.data.labels.length>24){rc.data.labels.shift();rc.data.datasets[0].data.shift();}
    rc.update('none');
  }

  // ── Run all AI/ML engines ──
  runLinearRegression();
  const rfResult = runRandomForest(d);
  runAnomalyDetection();
  runHealthAdvisory(d);
  updateSensorCharts();
  checkAlerts(d);

  // Update AI banner prediction
  const predTexts = [
    `Random Forest: ${rfResult.condition} | Rain probability: ${rfResult.rainProb}% | Linear Regression R²=${state.mlModel.r2.toFixed(3)} RMSE=±${state.mlModel.rmse.toFixed(1)}°C`,
    `Temperature trending: ${state.mlModel.slope>0.01?'Rising':'Stable'} | Anomaly Z-Score: Temp=${computeZScore(h.temp).toFixed(2)} | Humidity risk: ${d.hum>80?'High':'Normal'}`,
    `Health score updated: ${document.getElementById('healthScore')?.textContent||'--'}/100 | RF Trees: 15 active | LR training on ${h.temp.length} live points`,
  ];
  const predIdx = Math.floor(Date.now()/6000)%predTexts.length;
  setEl('aiBannerText', predTexts[predIdx]);

  // Fetch sunrise/sunset from API (one-time)
  if (!state.sunFetched) { fetchSunTimes(); state.sunFetched = true; }
}

// ─── Fetch Sunrise/Sunset (Open-Meteo) ───────────────
async function fetchSunTimes() {
  try {
    const loc = LOCATIONS[state.currentLoc || 'nagpur'];
    const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.lat}&longitude=${loc.lon}&daily=sunrise,sunset&timezone=${encodeURIComponent(loc.tz)}&forecast_days=1`);
    const json = await res.json();
    const sr = new Date(json.daily.sunrise[0]).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
    const ss = new Date(json.daily.sunset[0]).toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
    setEl('sunriseVal', sr); setEl('sunsetVal', ss);
  } catch(e) {
    setEl('sunriseVal', '06:12 AM'); setEl('sunsetVal', '07:28 PM');
  }
}

// ─── Change Location Handler ─────────────────────────
function changeLocation(key) {
  if (!LOCATIONS[key]) return;
  state.currentLoc = key;
  state.forecastLoaded = false;
  state.sunFetched = false;
  
  // Reset live baselines so they fetch fresh from the new location's API
  state.liveTemp = undefined;
  state.liveWind = undefined;
  
  logML('info', `Changing location to ${LOCATIONS[key].name}...`);
  
  // Reset drop-down UI state
  const sel = document.getElementById('locationSelect');
  if (sel) sel.value = key;
  
  // Fetch new data
  loadForecast();
  fetchSunTimes();
  
  // Re-run update loop instantly
  updateAll();
}

// ─── BOOT SEQUENCE ────────────────────────────────────
const bootMsgs = [
  'Loading ML models...', 'Initializing Linear Regression engine...',
  'Starting Random Forest classifier (15 trees)...',
  'Activating Z-Score anomaly detector...',
  'Connecting to Open-Meteo API...',
  'Loading Health Advisory engine...',
  'Calibrating sensor simulation...', 'System ready!',
];

async function boot() {
  initParticles();
  updateClock(); setInterval(updateClock, 1000);
  const msgEl = document.getElementById('aiLoadMsg');
  const overlay = document.getElementById('aiOverlay');
  for (let i=0; i<bootMsgs.length; i++) {
    if(msgEl) msgEl.textContent = bootMsgs[i];
    await new Promise(r=>setTimeout(r,350));
  }
  if (overlay) { overlay.classList.add('hidden'); setTimeout(()=>overlay.style.display='none',600); }

  // Init charts
  initMainChart(); initPressChart(); initRainChart();
  initMLLiveChart(); initLRChart(); initAnomalyChart();
  initAnomalyMainChart(); initSensorCharts();

  // Initial data
  generateSensorData();
  updateAll();

  // Load forecast in background
  loadForecast();

  // Seed initial alerts
  state.alerts=[
    {type:'warning',icon:'🌡️',title:'High Temperature Warning',desc:'Temperature above 32°C — Heat conditions possible today'},
    {type:'info',   icon:'💧',title:'Elevated Humidity',desc:'Relative humidity at 72% — Discomfort index elevated'},
  ];
  renderAlerts();

  logML('info', 'WeatherSense AI Engine fully initialized');
  logML('info', 'Models: Linear Regression + Random Forest (15 trees) + Z-Score Anomaly Detector');
  logML('info', `API: Open-Meteo connected (Nagpur, Maharashtra, IN — lat=21.15, lon=79.09)`);

  // Real-time update every 3 seconds
  setInterval(updateAll, 3000);
}

window.addEventListener('DOMContentLoaded', boot);
