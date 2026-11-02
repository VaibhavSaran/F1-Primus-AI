/* Shared pure clock projection: browser tower and coordinate markers use this snapshot. */
(function (root) {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  function validatePayload(p) {
    if (!p || p.schema_version !== 'replay-v1' || typeof p.session_id !== 'string' ||
        !/^replay-v1:[a-f0-9]{64}$/.test(p.frame_version || '') ||
        !Number.isInteger(p.duration_ms) || p.duration_ms < 0 || p.duration_ms > 86400000 ||
        !Array.isArray(p.entries) || !p.entries.length || p.entries.length > 64 ||
        !Array.isArray(p.laps) || p.laps.length > 20000 ||
        !Array.isArray(p.positions) || p.positions.length > 120000 ||
        !Array.isArray(p.markers) || p.markers.length > 50000 ||
        (p.weather !== undefined && (!Array.isArray(p.weather) || p.weather.length > 10000)) ||
        p.position_max_age_ms !== 5000 ||
        (p.weather !== undefined && p.weather_max_age_ms !== 120000) ||
        (p.timing !== undefined && (!Array.isArray(p.timing) || p.timing.length > 250000 ||
          p.timing_max_age_ms !== 120000 || !['timing_stream', 'lap_end_position', 'derived_timed_lap_record_order'].includes(p.timing_basis))) ||
        new TextEncoder().encode(JSON.stringify(p)).length > 20000000) return false;
    const ids = new Set(p.entries.map(e => e.driver_id));
    if (ids.size !== p.entries.length || [...ids].some(d => typeof d !== 'string')) return false;
    const clock = r => Number.isInteger(r.clock_ms) && r.clock_ms >= 0 && r.clock_ms <= p.duration_ms;
    const gap = v => Array.isArray(v) && v.length === 3 &&
      (v[0] === null || finite(v[0]) && v[0] >= 0) &&
      (v[1] === null || Number.isInteger(v[1]) && v[1] > 0) &&
      !(v[0] !== null && v[1] !== null) && (v[2] === null || typeof v[2] === 'string' && v[2].length <= 2048);
    const sorted = rows => rows.every((r, i) => r && typeof r === 'object' &&
      (!i || rows[i - 1].clock_ms <= r.clock_ms));
    if (![p.laps, p.positions, p.markers, p.weather || [], p.timing || []].every(sorted)) return false;
    if (!(p.timing || []).every(r => ids.has(r.driver_id) && clock(r) &&
      (!('position' in r) || r.position === null || Number.isInteger(r.position) && r.position >= 1 && r.position <= 64) &&
      (!('gap' in r) || gap(r.gap)) && (!('interval' in r) || gap(r.interval)))) return false;
    return p.laps.every(r => ids.has(r.driver_id) && clock(r) && Number.isInteger(r.lap_number) && r.lap_number > 0) &&
      p.positions.every(r => ids.has(r.driver_id) && clock(r) && finite(r.x) && finite(r.y)) &&
      p.markers.every(r => clock(r) && typeof r.message === 'string') &&
      (p.weather || []).every(r => clock(r) && ['air_temperature_c', 'track_temperature_c',
        'relative_humidity_pct', 'pressure_hpa', 'wind_speed_mps', 'wind_direction_deg', 'rainfall_flag']
        .every(k => r[k] === null || finite(r[k])) && [null, 0, 1].includes(r.rainfall_flag));
  }
  function frameAt(p, clockMs) {
    const latest = new Map(), positions = new Map(), counts = new Map();
    const timing = new Map(), basis = p.timing_basis || 'derived_timed_lap_record_order';
    for (const row of p.timing || []) {
      if (row.clock_ms > clockMs) break;
      if (!timing.has(row.driver_id)) timing.set(row.driver_id, {});
      for (const field of ['position', 'gap', 'interval']) {
        if (field in row) timing.get(row.driver_id)[field] = [row.clock_ms, row[field]];
      }
    }
    for (const lap of p.laps) {
      if (lap.clock_ms > clockMs) break;
      if (finite(lap.lap_time_s) && lap.lap_time_s > 0) {
        latest.set(lap.driver_id, lap);
        counts.set(lap.driver_id, (counts.get(lap.driver_id) || 0) + 1);
      }
    }
    for (const sample of p.positions) {
      if (sample.clock_ms > clockMs) break;
      positions.set(sample.driver_id, sample);
    }
    const drivers = p.entries.map(entry => {
      const lap = latest.get(entry.driver_id), sample = positions.get(entry.driver_id);
      const observed = timing.get(entry.driver_id) || {};
      const fields = {position: null, gap_s: null, gap_laps: null, gap_label: null,
        interval_s: null, interval_laps: null, interval_label: null, timing_basis: basis};
      for (const field of ['position', 'gap', 'interval']) {
        const value = observed[field], fresh = !!value && clockMs - value[0] <= p.timing_max_age_ms;
        fields[`${field}_fresh`] = fresh;
        if (fresh) {
          if (field === 'position') fields.position = value[1];
          else [fields[`${field}_s`], fields[`${field}_laps`], fields[`${field}_label`]] = value[1];
        }
      }
      for (const field of ['gap', 'interval']) {
        if (fields[`${field}_label`] === 'leader_counter') fields[`${field}_label`] = field === 'gap' && fields.position === 1 ? 'Leader' : null;
      }
      fields.timing_fresh = fields.position_fresh || fields.gap_fresh || fields.interval_fresh;
      return {...entry, completed_laps: counts.get(entry.driver_id) || 0,
        last_lap_number: lap ? lap.lap_number : null,
        last_lap_time_s: lap ? lap.lap_time_s : null, last_lap_clock_ms: lap ? lap.clock_ms : null,
        compound: lap ? lap.compound : null,
        tyre_age_laps: lap ? lap.tyre_age_laps : null, ...fields,
        coordinates: sample && clockMs - sample.clock_ms <= p.position_max_age_ms ?
          {x: sample.x, y: sample.y, sample_clock_ms: sample.clock_ms} : null};
    });
    drivers.sort((a, b) => b.completed_laps - a.completed_laps ||
      (a.last_lap_clock_ms ?? Infinity) - (b.last_lap_clock_ms ?? Infinity) || a.driver_id.localeCompare(b.driver_id));
    drivers.forEach((d, i) => { d.completion_order = d.completed_laps ? i + 1 : null; });
    if (basis !== 'derived_timed_lap_record_order') drivers.sort((a, b) =>
      (a.position ?? Infinity) - (b.position ?? Infinity) || a.driver_id.localeCompare(b.driver_id));
    const latestWeather = (p.weather || []).filter(w => w.clock_ms <= clockMs).at(-1);
    const weather = latestWeather && clockMs - latestWeather.clock_ms <= p.weather_max_age_ms ? latestWeather : {};
    return {clock_ms: clockMs, drivers, weather, markers: p.markers.filter(m => m.clock_ms <= clockMs)};
  }
  function acceptRender(event, parentSource, expectedOrigin) {
    return event.source === parentSource && event.origin === expectedOrigin &&
      event.data?.type === 'streamlit:render' && !!event.data.args;
  }
  function trackBounds(positions) {
    if (!positions.length) return null;
    const bounds = {xMin: Infinity, xMax: -Infinity, yMin: Infinity, yMax: -Infinity};
    for (const p of positions) {
      bounds.xMin = Math.min(bounds.xMin, p.x); bounds.xMax = Math.max(bounds.xMax, p.x);
      bounds.yMin = Math.min(bounds.yMin, p.y); bounds.yMax = Math.max(bounds.yMax, p.y);
    }
    if (bounds.xMax === bounds.xMin) { bounds.xMin -= 1; bounds.xMax += 1; }
    if (bounds.yMax === bounds.yMin) { bounds.yMin -= 1; bounds.yMax += 1; }
    return bounds;
  }
  function formatClock(ms) {
    const seconds = Math.floor(ms / 1000);
    return `${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
  }
  const api = {validatePayload, frameAt, acceptRender, trackBounds, formatClock};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReplayClock = api;
})(globalThis);
