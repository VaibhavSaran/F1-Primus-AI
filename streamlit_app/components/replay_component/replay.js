/* Streamlit custom-component v1 protocol; all assets are local and no frames call Python. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const parentOrigin = new URL(document.referrer || location.href).origin;
  let payload = null, clockMs = 0, playing = false, sequence = 0, revision = null;
  let selected = [], previousTime = null, renderTime = -Infinity;
  let trackBounds = null, background = null;
  const label = driver => payload.driver_labels?.[driver.driver_id] || driver.abbreviation || 'Unknown driver';
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const post = (type, data = {}) => window.parent.postMessage({isStreamlitMessage: true, type, ...data}, parentOrigin);
  const size = () => post('streamlit:setFrameHeight', {height: document.documentElement.scrollHeight});
  function stop() { playing = false; previousTime = null; $('play').textContent = 'Play'; }
  function commit(event) {
    if (!payload) return;
    stop(); sequence += 1;
    post('streamlit:setComponentValue', {dataType: 'json', value: {
      session_id: payload.session_id, frame_version: payload.frame_version,
      sequence, clock_ms: Math.round(clockMs), driver_ids: selected, event}});
  }
  function seek(value) {
    if (!payload) return;
    clockMs = Math.max(0, Math.min(payload.duration_ms, Math.round(value)));
    stop(); draw(); commit('seek');
  }
  function drawTrack(frame) {
    const canvas = $('track'), context = canvas.getContext('2d');
    const width = Math.max(1, Math.round(canvas.clientWidth || canvas.width));
    const height = Math.max(1, Math.round(canvas.clientHeight || canvas.height));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height; background = null;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    const known = frame.drivers.filter(d => d.coordinates);
    if (!trackBounds) {
      context.fillStyle = '#a8b4c4'; context.font = '16px system-ui';
      context.fillText('No fresh recorded coordinates at this clock', 20, 150); return;
    }
    const {xMin, xMax, yMin, yMax} = trackBounds;
    const scale = Math.min((canvas.width - 70) / Math.max(1, xMax - xMin), (canvas.height - 60) / Math.max(1, yMax - yMin));
    const left = (canvas.width - (xMax - xMin) * scale) / 2;
    const bottom = (canvas.height + (yMax - yMin) * scale) / 2;
    const project = p => [left + (p.x - xMin) * scale, bottom - (p.y - yMin) * scale];
    if (!background) {
      background = document.createElement('canvas');
      background.width = canvas.width; background.height = canvas.height;
      const ctx = background.getContext('2d'), previous = new Map();
      ctx.strokeStyle = '#526373'; ctx.lineWidth = 2; ctx.beginPath();
      for (const p of payload.positions) {
        const [x, y] = project(p), last = previous.get(p.driver_id);
        if (!last || p.clock_ms - last.clock_ms > Math.max(5000, payload.position_max_age_ms)) ctx.moveTo(x, y);
        else { ctx.moveTo(...project(last)); ctx.lineTo(x, y); }
        previous.set(p.driver_id, p);
      }
      ctx.stroke();
    }
    if (background) context.drawImage(background, 0, 0);
    for (const driver of known) {
      const [x, y] = project(driver.coordinates);
      const palette = ['#73e5f2', '#ffae70', '#d2a0ff', '#b6e778', '#ff91ad'];
      let hash = 0;
      for (const char of driver.driver_id) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
      context.fillStyle = palette[hash % palette.length];
      context.beginPath(); context.arc(x, y, 5, 0, Math.PI * 2); context.fill();
      if (selected.includes(driver.driver_id)) { context.strokeStyle = '#ffffff'; context.lineWidth = 2; context.stroke(); }
      context.font = '14px system-ui'; context.fillText(driver.abbreviation || label(driver), x + 8, y - 7);
    }
  }
  function draw() {
    if (!payload) return;
    const frame = ReplayClock.frameAt(payload, Math.round(clockMs));
    $('clock').textContent = ReplayClock.formatClock(frame.clock_ms);
    $('seek').value = String(frame.clock_ms);
    const official = payload.timing_basis === 'timing_stream';
    const checkpoint = payload.timing_basis === 'lap_end_position';
    $('tower-title').textContent = official ? 'Race standings' : checkpoint ?
      'Lap-end classification checkpoints' : 'Session lap completion';
    $('tower-note').textContent = official ?
      'Recorded timing snapshots carry the last reported values; missing values and snapshots older than two minutes show —. Interval refers to the car ahead in the full field.' : checkpoint ?
      'Position is the latest recorded lap-end checkpoint, not live race order. Gaps and intervals are unavailable between checkpoints.' :
      'Ordered by positively timed lap records and their clock, not race order. Missing durations do not certify completed laps; running positions and gaps are unavailable.';
    const columns = official ? ['Position', 'Driver', 'Gap to leader', 'Interval', 'Last lap (s)', 'Tyres'] :
      checkpoint ? ['Position', 'Driver', 'Last lap (s)', 'Tyres'] :
      ['Order', 'Driver', 'Timed laps observed', 'Last lap (s)', 'Tyres'];
    $('tower-head').replaceChildren(...columns.map(name => {
      const th = document.createElement('th'); th.textContent = name; th.setAttribute('scope', 'col'); return th;
    }));
    const tower = $('tower');
    while (tower.children.length > frame.drivers.length) tower.removeChild(tower.lastChild);
    for (const [index, driver] of frame.drivers.entries()) {
      const tr = tower.children[index] || document.createElement('tr'), nameCell = document.createElement('td');
      tr.replaceChildren();
      const append = value => { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); };
      if (official || checkpoint) append(driver.position ?? '—');
      else append(index + 1);
      const button = document.createElement('button');
      button.textContent = label(driver);
      button.setAttribute('aria-label', `Select driver ${label(driver)} and update charts`);
      button.setAttribute('aria-pressed', String(selected.includes(driver.driver_id)));
      button.onclick = () => { selected = [driver.driver_id]; commit('selection'); draw(); };
      nameCell.append(button); tr.append(nameCell);
      if (official) {
        const gap = driver.gap_label === 'Leader' && !(driver.position === 1 && driver.position_fresh) ? null : driver.gap_label;
        append(gap ?? '—'); append(driver.interval_label === 'Leader' ? '—' : driver.interval_label ?? '—');
      } else if (!checkpoint) {
        append(driver.completed_laps);
      }
      append(driver.last_lap_time_s?.toFixed(3) ?? 'Unknown'); append(driver.compound ?? 'Unknown');
      if (!tower.children[index]) tower.append(tr);
    }
    const last = frame.markers.at(-1);
    $('marker').textContent = last ? `${ReplayClock.formatClock(last.clock_ms)} · ${last.message}` : 'No observed pit / flag marker by this clock.';
    const w = frame.weather;
    const value = (key, unit) => w[key] == null ? 'Unknown' : `${w[key].toFixed(1)}${unit}`;
    $('weather-air').textContent = value('air_temperature_c', ' °C');
    $('weather-track').textContent = value('track_temperature_c', ' °C');
    $('weather-humidity').textContent = value('relative_humidity_pct', '%');
    $('weather-pressure').textContent = value('pressure_hpa', ' hPa');
    $('weather-wind').textContent = `${value('wind_speed_mps', ' m/s')} / ${value('wind_direction_deg', '°')}`;
    $('weather-rain').textContent = w.rainfall_flag == null ? 'Unknown' : w.rainfall_flag ? 'Yes' : 'No';
    $('weather-note').textContent = w.clock_ms == null ? 'No fresh weather observation (older than 2 min or unavailable).' : `Weather observed ${ReplayClock.formatClock(w.clock_ms)}; fields without readings stay unknown.`;
    drawTrack(frame);
  }
  $('play').onclick = () => {
    if (!payload) return;
    if (playing) { commit('pause'); return; }
    if (reducedMotion.matches) { $('state').textContent = 'Reduced motion is active. Use native seek or the browser seek slider.'; return; }
    if (clockMs >= payload.duration_ms) clockMs = 0;
    playing = true; previousTime = null; $('play').textContent = 'Pause';
  };
  $('back').onclick = () => seek(clockMs - 10000);
  $('forward').onclick = () => seek(clockMs + 10000);
  $('seek').oninput = event => { stop(); clockMs = Number(event.target.value); draw(); };
  $('seek').onchange = () => commit('seek');
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches && playing) commit('pause'); });
  document.addEventListener('visibilitychange', () => { if (document.hidden && playing) commit('pause'); });
  window.addEventListener('message', event => {
    if (!ReplayClock.acceptRender(event, window.parent, parentOrigin)) return;
    const args = event.data.args, incoming = args.payload;
    if (!ReplayClock.validatePayload(incoming) || !Number.isInteger(args.committed_clock_ms) ||
        args.committed_clock_ms < 0 || args.committed_clock_ms > incoming.duration_ms ||
        !Number.isInteger(args.last_sequence) || !Number.isInteger(args.commit_revision) ||
        !Array.isArray(args.selected_driver_ids) || !args.selected_driver_ids.length || args.selected_driver_ids.length > incoming.entries.length ||
        new Set(args.selected_driver_ids).size !== args.selected_driver_ids.length ||
        args.selected_driver_ids.some(d => !incoming.entries.some(e => e.driver_id === d))) {
      stop(); $('state').textContent = 'Invalid replay payload. Use the native controls.'; return;
    }
    const changed = !payload || incoming.frame_version !== payload.frame_version || incoming.session_id !== payload.session_id;
    payload = incoming;
    if (changed) {
      trackBounds = ReplayClock.trackBounds(payload.positions);
      background = null;
    }
    if (changed || revision !== args.commit_revision) {
      stop(); clockMs = args.committed_clock_ms; selected = args.selected_driver_ids;
      revision = args.commit_revision;
    }
    sequence = Math.max(sequence, args.last_sequence);
    $('seek').max = String(payload.duration_ms);
    $('state').textContent = 'Recorded session playback';
    $('track-note').textContent = payload.track_unavailable_reason ||
      'Full-session observed paths provide the background, including recorded pit-lane paths. Markers follow playback time and disappear when recordings are stale. Colors identify drivers.';
    draw(); size();
  });
  function animate(now) {
    if (playing && payload) {
      if (previousTime !== null) clockMs = Math.min(payload.duration_ms, clockMs + (now - previousTime) * Number($('speed').value));
      previousTime = now;
      // Bound DOM/canvas work; this is not a measured FPS guarantee.
      if (now - renderTime >= 30) { draw(); renderTime = now; }
      if (clockMs >= payload.duration_ms) { draw(); commit('pause'); }
    }
    requestAnimationFrame(animate);
  }
  new ResizeObserver(() => { if (payload) draw(); size(); }).observe(document.body);
  post('streamlit:componentReady', {apiVersion: 1}); size(); requestAnimationFrame(animate);
})();
