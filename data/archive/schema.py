"""Local historical archive schema. Units and clocks are explicit in column names."""

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_objects (
 source_id TEXT PRIMARY KEY NOT NULL, provider TEXT NOT NULL, request_fingerprint TEXT NOT NULL,
 endpoint_kind TEXT, canonical_url TEXT, first_fetched_at_utc TEXT NOT NULL,
 original_published_at_utc TEXT, original_issued_at_utc TEXT, original_available_at_utc TEXT,
 availability_basis TEXT, payload_sha256 TEXT NOT NULL, raw_path TEXT NOT NULL,
 source_schema_version TEXT, quality_flags_json TEXT,
 UNIQUE(provider,request_fingerprint,payload_sha256));
CREATE TABLE IF NOT EXISTS ingestion_runs (
 ingestion_run_id TEXT PRIMARY KEY NOT NULL, script_version TEXT, parser_version TEXT,
 schema_version TEXT, config_hash TEXT, start_utc TEXT, end_utc TEXT,
 status TEXT, manifest_path TEXT);
CREATE TABLE IF NOT EXISTS seasons (
 year INTEGER PRIMARY KEY, calendar_status TEXT, source_id TEXT REFERENCES source_objects,
 last_verified_at_utc TEXT);
CREATE TABLE IF NOT EXISTS circuits (
 circuit_id TEXT PRIMARY KEY NOT NULL, name TEXT, country TEXT, locality TEXT,
 reference_latitude REAL, reference_longitude REAL, elevation_m REAL, timezone_name TEXT);
CREATE TABLE IF NOT EXISTS circuit_layouts (
 layout_id TEXT PRIMARY KEY NOT NULL, circuit_id TEXT REFERENCES circuits, layout_name TEXT,
 valid_from TEXT, valid_to TEXT, length_m REAL, rotation_deg REAL,
 coordinate_frame_metadata_json TEXT, provenance_source_id TEXT REFERENCES source_objects);
CREATE TABLE IF NOT EXISTS events (
 event_id TEXT PRIMARY KEY NOT NULL, season INTEGER NOT NULL REFERENCES seasons(year),
 circuit_id TEXT REFERENCES circuits, layout_id TEXT REFERENCES circuit_layouts,
 event_name TEXT, official_name TEXT, round_number INTEGER NOT NULL,
 event_format_raw TEXT, local_race_date TEXT, country TEXT, location TEXT, status TEXT,
 calendar_source_id TEXT REFERENCES source_objects, UNIQUE(season,round_number));
CREATE TABLE IF NOT EXISTS sessions (
 session_id TEXT PRIMARY KEY NOT NULL, event_id TEXT NOT NULL REFERENCES events,
 session_code TEXT NOT NULL, full_name TEXT, original_provider_name TEXT,
 scheduled_start_utc_ns INTEGER, source_timezone_name TEXT, status TEXT,
 schedule_source_id TEXT REFERENCES source_objects, UNIQUE(event_id,session_code));
CREATE TABLE IF NOT EXISTS drivers (
 driver_id TEXT PRIMARY KEY NOT NULL, first_name TEXT, last_name TEXT, display_name TEXT,
 country_code TEXT, source_driver_id TEXT);
CREATE TABLE IF NOT EXISTS teams (
 team_id TEXT PRIMARY KEY NOT NULL, canonical_name TEXT, source_team_id TEXT);
CREATE TABLE IF NOT EXISTS session_versions (
 version_id TEXT PRIMARY KEY NOT NULL, session_id TEXT NOT NULL REFERENCES sessions,
 ingestion_run_id TEXT REFERENCES ingestion_runs, created_at_utc TEXT NOT NULL,
 normalizer_version TEXT NOT NULL, dataset_hash TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('staged','published','rejected')),
 observed_start_utc_ns INTEGER, observed_end_utc_ns INTEGER, t0_utc_ns INTEGER,
 completeness_json TEXT, quality_flags_json TEXT, metadata_json TEXT,
 UNIQUE(session_id,dataset_hash));
CREATE TABLE IF NOT EXISTS session_version_sources (
 version_id TEXT NOT NULL REFERENCES session_versions, source_id TEXT NOT NULL REFERENCES source_objects,
 dataset_kind TEXT NOT NULL, PRIMARY KEY(version_id,source_id,dataset_kind));
CREATE TABLE IF NOT EXISTS session_entries (
 version_id TEXT NOT NULL REFERENCES session_versions, driver_id TEXT NOT NULL REFERENCES drivers,
 team_id TEXT REFERENCES teams, driver_number TEXT, abbreviation TEXT, broadcast_name TEXT,
 team_name_at_event TEXT, team_color_at_event TEXT, source_driver_id TEXT, source_team_id TEXT,
 PRIMARY KEY(version_id,driver_id), UNIQUE(version_id,driver_number));
CREATE TABLE IF NOT EXISTS session_results (
 version_id TEXT NOT NULL, driver_id TEXT NOT NULL, finishing_position REAL,
 classified_position_raw TEXT, grid_position REAL, q1_ns INTEGER, q2_ns INTEGER,
 q3_ns INTEGER, reported_time_ns INTEGER, time_semantics TEXT, points REAL,
 laps_completed REAL, status_raw TEXT, normalized_outcome_status TEXT,
 original_publication_utc TEXT, PRIMARY KEY(version_id,driver_id),
 FOREIGN KEY(version_id,driver_id) REFERENCES session_entries(version_id,driver_id));
CREATE TABLE IF NOT EXISTS laps (
 version_id TEXT NOT NULL, driver_id TEXT NOT NULL, lap_number INTEGER NOT NULL CHECK(lap_number>0),
 stint_number INTEGER, lap_start_clock_ns INTEGER, lap_end_clock_ns INTEGER, lap_start_utc_ns INTEGER,
 lap_time_ns INTEGER, sector1_ns INTEGER, sector2_ns INTEGER, sector3_ns INTEGER,
 sector1_end_clock_ns INTEGER, sector2_end_clock_ns INTEGER, sector3_end_clock_ns INTEGER,
 speed_i1_kmh REAL, speed_i2_kmh REAL, speed_finish_line_kmh REAL, speed_trap_kmh REAL,
 pit_in_clock_ns INTEGER, pit_out_clock_ns INTEGER, compound_raw TEXT, compound_normalized TEXT,
 tyre_age_laps REAL, fresh_tyre INTEGER CHECK(fresh_tyre IN(0,1)),
 is_personal_best INTEGER CHECK(is_personal_best IN(0,1)), track_status_raw TEXT,
 lap_position REAL, position_basis TEXT, deleted INTEGER CHECK(deleted IN(0,1)),
 deleted_reason TEXT, provider_is_accurate INTEGER CHECK(provider_is_accurate IN(0,1)),
 fastf1_generated INTEGER CHECK(fastf1_generated IN(0,1)), lap_quality_flags_json TEXT,
 PRIMARY KEY(version_id,driver_id,lap_number),
 FOREIGN KEY(version_id,driver_id) REFERENCES session_entries(version_id,driver_id));
CREATE TABLE IF NOT EXISTS circuit_markers (
 version_id TEXT NOT NULL REFERENCES session_versions, marker_kind TEXT NOT NULL,
 marker_index INTEGER NOT NULL CHECK(marker_index>=0), layout_id TEXT REFERENCES circuit_layouts,
 kind TEXT, number INTEGER, letter TEXT, x_raw REAL, y_raw REAL, angle_deg REAL,
 distance_m REAL, geometry_basis TEXT, source_id TEXT REFERENCES source_objects,
 PRIMARY KEY(version_id,marker_kind,marker_index));
CREATE TABLE IF NOT EXISTS track_status_events (
 version_id TEXT NOT NULL REFERENCES session_versions, event_index INTEGER NOT NULL CHECK(event_index>=0),
 session_clock_ns INTEGER, timestamp_utc_ns INTEGER, status_code_raw TEXT, message_raw TEXT,
 PRIMARY KEY(version_id,event_index));
CREATE TABLE IF NOT EXISTS session_status_events (
 version_id TEXT NOT NULL REFERENCES session_versions, event_index INTEGER NOT NULL CHECK(event_index>=0),
 session_clock_ns INTEGER, timestamp_utc_ns INTEGER, status_raw TEXT,
 PRIMARY KEY(version_id,event_index));
CREATE TABLE IF NOT EXISTS race_control_messages (
 version_id TEXT NOT NULL REFERENCES session_versions, message_index INTEGER NOT NULL CHECK(message_index>=0),
 timestamp_utc_ns INTEGER, session_clock_ns INTEGER, category_raw TEXT, message_text TEXT,
 status_raw TEXT, flag_raw TEXT, scope_raw TEXT, sector INTEGER, driver_number_raw TEXT,
 driver_id TEXT REFERENCES drivers, lap_number INTEGER, PRIMARY KEY(version_id,message_index));
CREATE TABLE IF NOT EXISTS pit_stop_records (
 version_id TEXT NOT NULL, driver_id TEXT NOT NULL, source_id TEXT NOT NULL REFERENCES source_objects,
 stop_index INTEGER NOT NULL, duration_ns INTEGER, duration_kind TEXT NOT NULL,
 source_evidence_json TEXT, PRIMARY KEY(version_id,driver_id,source_id,stop_index),
 FOREIGN KEY(version_id,driver_id) REFERENCES session_entries(version_id,driver_id));
CREATE TABLE IF NOT EXISTS weather_series (
 series_id TEXT PRIMARY KEY NOT NULL, circuit_id TEXT REFERENCES circuits,
 session_version_id TEXT REFERENCES session_versions, source_id TEXT REFERENCES source_objects,
 provider TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN('track_observation','reanalysis','forecast')),
 model_name TEXT, requested_latitude REAL, requested_longitude REAL, returned_latitude REAL,
 returned_longitude REAL, coordinate_basis TEXT, elevation_m REAL, spatial_resolution_km REAL,
 original_timezone TEXT, units_metadata_json TEXT, aggregation_metadata_json TEXT,
 dataset_version TEXT, quality_flags_json TEXT);
CREATE TABLE IF NOT EXISTS weather_observations (
 series_id TEXT NOT NULL REFERENCES weather_series, sample_index INTEGER NOT NULL CHECK(sample_index>=0),
 valid_time_utc_ns INTEGER, session_clock_ns INTEGER, air_temperature_c REAL,
 track_temperature_c REAL, relative_humidity_pct REAL, pressure_hpa REAL, pressure_kind TEXT,
 wind_speed_mps REAL, wind_direction_deg REAL, wind_gust_mps REAL,
 rainfall_flag INTEGER CHECK(rainfall_flag IN(0,1)), rain_mm REAL, precipitation_mm REAL,
 cloud_cover_pct REAL, dew_point_c REAL, weather_code INTEGER,
 precipitation_interval_start_utc_ns INTEGER, precipitation_interval_end_utc_ns INTEGER,
 source_id TEXT REFERENCES source_objects, quality_flags TEXT, PRIMARY KEY(series_id,sample_index));
CREATE TRIGGER IF NOT EXISTS observation_kind BEFORE INSERT ON weather_observations
 WHEN (SELECT kind FROM weather_series WHERE series_id=NEW.series_id)='forecast'
 BEGIN SELECT RAISE(ABORT,'forecast series cannot contain observations'); END;
CREATE TABLE IF NOT EXISTS weather_forecast_runs (
 forecast_run_id TEXT PRIMARY KEY NOT NULL, series_id TEXT NOT NULL REFERENCES weather_series,
 source_id TEXT REFERENCES source_objects, model_run_id TEXT, initialized_at_utc TEXT,
 issued_at_utc TEXT, known_available_at_utc TEXT, fetched_at_utc TEXT,
 archived_run_identifier TEXT, availability_basis TEXT,
 historical_publication_verified INTEGER CHECK(historical_publication_verified IN(0,1)),
 forecast_horizon_start_utc_ns INTEGER, forecast_horizon_end_utc_ns INTEGER, raw_payload_hash TEXT);
CREATE TRIGGER IF NOT EXISTS forecast_kind BEFORE INSERT ON weather_forecast_runs
 WHEN (SELECT kind FROM weather_series WHERE series_id=NEW.series_id)!='forecast'
 BEGIN SELECT RAISE(ABORT,'forecast run requires forecast series'); END;
CREATE TABLE IF NOT EXISTS weather_forecast_values (
 forecast_run_id TEXT NOT NULL REFERENCES weather_forecast_runs, valid_time_utc_ns INTEGER NOT NULL,
 lead_time_s REAL, air_temperature_c REAL, relative_humidity_pct REAL, pressure_hpa REAL,
 pressure_kind TEXT, wind_speed_mps REAL, wind_direction_deg REAL, wind_gust_mps REAL,
 precipitation_probability_pct REAL, precipitation_mm REAL, rain_mm REAL,
 cloud_cover_pct REAL, weather_code INTEGER, precipitation_interval_start_utc_ns INTEGER,
 precipitation_interval_end_utc_ns INTEGER, quality_flags TEXT,
 PRIMARY KEY(forecast_run_id,valid_time_utc_ns));
CREATE TABLE IF NOT EXISTS collection_tasks (
 task_id TEXT PRIMARY KEY NOT NULL, requested_scope TEXT NOT NULL, dataset_kind TEXT NOT NULL,
 config_version TEXT NOT NULL, season INTEGER, event_id TEXT REFERENCES events,
 session_id TEXT REFERENCES sessions, status TEXT NOT NULL
 CHECK(status IN('pending','running','complete','partial','blocked','unavailable','failed')),
 attempt_count INTEGER NOT NULL DEFAULT 0, next_retry_utc TEXT, checkpoint_json TEXT,
 last_error_code TEXT, expected_count INTEGER, observed_count INTEGER,
 UNIQUE(requested_scope,dataset_kind,config_version));
CREATE TABLE IF NOT EXISTS fetch_attempts (
 attempt_id TEXT PRIMARY KEY NOT NULL, task_id TEXT REFERENCES collection_tasks,
 source_id TEXT REFERENCES source_objects, provider TEXT NOT NULL, canonical_request_hash TEXT,
 requested_at_utc TEXT, completed_at_utc TEXT, response_status INTEGER,
 from_cache INTEGER CHECK(from_cache IN(0,1)), bytes_received INTEGER,
 retry_after_utc TEXT, error_type TEXT);
CREATE TABLE IF NOT EXISTS dataset_files (
 file_id TEXT PRIMARY KEY NOT NULL, version_id TEXT REFERENCES session_versions,
 source_id TEXT REFERENCES source_objects, kind TEXT NOT NULL, driver_id TEXT REFERENCES drivers,
 relative_path TEXT NOT NULL UNIQUE, file_sha256 TEXT NOT NULL, schema_version TEXT,
 row_count INTEGER NOT NULL CHECK(row_count>=0), first_clock_ns INTEGER, last_clock_ns INTEGER,
 first_utc_ns INTEGER, last_utc_ns INTEGER, compression TEXT, transform_version TEXT,
 published_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS coverage (
 version_id TEXT NOT NULL REFERENCES session_versions, dataset_kind TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN('available','partial','unavailable','unsupported','blocked')),
 rows INTEGER, valid_count INTEGER, invalid_count INTEGER, missing_columns_json TEXT,
 reason TEXT, checked_at_utc TEXT, PRIMARY KEY(version_id,dataset_kind));
CREATE TABLE IF NOT EXISTS replay_builds (
 replay_build_id TEXT PRIMARY KEY NOT NULL, session_version_id TEXT NOT NULL REFERENCES session_versions,
 source_manifest_hash TEXT, derivation_version TEXT, playback_origin_session_clock_ns INTEGER,
 frame_interval_ns INTEGER, interpolation_policy TEXT, discrete_channel_policy TEXT,
 maximum_gap_ns INTEGER, output_manifest_json TEXT, created_at_utc TEXT, validation_status TEXT);
CREATE INDEX IF NOT EXISTS laps_cross_driver ON laps(version_id,lap_number,driver_id);
CREATE INDEX IF NOT EXISTS laps_clock ON laps(version_id,driver_id,lap_end_clock_ns);
CREATE INDEX IF NOT EXISTS laps_stint ON laps(version_id,driver_id,stint_number);
CREATE INDEX IF NOT EXISTS weather_time ON weather_observations(series_id,valid_time_utc_ns);
CREATE INDEX IF NOT EXISTS forecast_available ON weather_forecast_runs(series_id,known_available_at_utc);
CREATE INDEX IF NOT EXISTS messages_clock ON race_control_messages(version_id,session_clock_ns,message_index);
CREATE INDEX IF NOT EXISTS dataset_clock ON dataset_files(version_id,kind,driver_id,first_clock_ns,last_clock_ns);
CREATE INDEX IF NOT EXISTS versions_session ON session_versions(session_id,status,created_at_utc);
CREATE VIEW IF NOT EXISTS v_stints AS
 SELECT version_id,driver_id,stint_number,min(lap_number) first_observed_lap,
 max(lap_number) last_observed_lap,count(*) observed_lap_count,
 min(tyre_age_laps) start_tyre_age_laps,max(tyre_age_laps) end_tyre_age_laps,
 CASE WHEN count(DISTINCT compound_raw)=1 THEN min(compound_raw) END compound_raw,
 max(lap_number)-min(lap_number)+1-count(*) missing_lap_count
 FROM laps WHERE stint_number IS NOT NULL GROUP BY version_id,driver_id,stint_number;
CREATE VIEW IF NOT EXISTS v_pit_visits AS
 WITH paired AS (SELECT i.version_id,i.driver_id,i.lap_number entry_lap,i.pit_in_clock_ns,
 (SELECT min(o.pit_out_clock_ns) FROM laps o WHERE o.version_id=i.version_id
 AND o.driver_id=i.driver_id AND o.pit_out_clock_ns>=i.pit_in_clock_ns
 AND NOT EXISTS(SELECT 1 FROM laps n WHERE n.version_id=i.version_id AND n.driver_id=i.driver_id
 AND n.pit_in_clock_ns>i.pit_in_clock_ns AND n.pit_in_clock_ns<o.pit_out_clock_ns)) pit_out_clock_ns,
 'pit_lane_visit' duration_kind FROM laps i WHERE i.pit_in_clock_ns IS NOT NULL)
 SELECT p.*,o.lap_number exit_lap,p.pit_out_clock_ns-p.pit_in_clock_ns duration_ns,
 p.pit_out_clock_ns IS NULL unmatched_entry,0 unmatched_exit FROM paired p
 LEFT JOIN laps o ON o.version_id=p.version_id AND o.driver_id=p.driver_id
 AND o.pit_out_clock_ns=p.pit_out_clock_ns
 UNION ALL
 SELECT o.version_id,o.driver_id,NULL,NULL,o.pit_out_clock_ns,'pit_lane_visit',
 o.lap_number,NULL,0,1 FROM laps o WHERE o.pit_out_clock_ns IS NOT NULL
 AND NOT EXISTS(SELECT 1 FROM paired p WHERE p.version_id=o.version_id
 AND p.driver_id=o.driver_id AND p.pit_out_clock_ns=o.pit_out_clock_ns);
CREATE VIEW IF NOT EXISTS v_session_driver_summary AS
 WITH clean AS (SELECT *,row_number() OVER(PARTITION BY version_id,driver_id ORDER BY lap_time_ns) rn,
 count(*) OVER(PARTITION BY version_id,driver_id) n FROM laps WHERE lap_time_ns>0
 AND provider_is_accurate=1 AND deleted=0 AND coalesce(fastf1_generated,0)=0
 AND pit_in_clock_ns IS NULL AND pit_out_clock_ns IS NULL AND track_status_raw='1')
 SELECT version_id,driver_id,count(*) clean_timed_lap_count,min(lap_time_ns) best_lap_ns,
 avg(lap_time_ns) mean_lap_ns,avg(CASE WHEN rn IN((n+1)/2,(n+2)/2) THEN lap_time_ns END) median_lap_ns,
 max(lap_time_ns)-min(lap_time_ns) lap_range_ns,
 'accurate,not_deleted,not_generated,no_pit,track_status_1,positive_time' clean_lap_definition
 FROM clean GROUP BY version_id,driver_id;
CREATE VIEW IF NOT EXISTS v_session_weather AS
 SELECT s.session_version_id,w.*,s.provider,s.kind,s.circuit_id FROM weather_observations w
 JOIN weather_series s USING(series_id);
CREATE VIEW IF NOT EXISTS v_lap_weather_asof_start AS
 SELECT l.version_id,l.driver_id,l.lap_number,s.series_id,w.sample_index,
 l.lap_start_utc_ns-w.valid_time_utc_ns age_ns,300000000000 maximum_age_ns,s.kind
 FROM laps l JOIN session_versions v USING(version_id) JOIN sessions ss USING(session_id)
 JOIN events e ON e.event_id=ss.event_id JOIN weather_series s
 ON (s.session_version_id=l.version_id OR(s.session_version_id IS NULL AND s.circuit_id=e.circuit_id))
 JOIN weather_observations w ON w.series_id=s.series_id
 AND w.valid_time_utc_ns<=l.lap_start_utc_ns AND w.valid_time_utc_ns>=l.lap_start_utc_ns-300000000000
 WHERE w.sample_index=(SELECT w2.sample_index FROM weather_observations w2
 WHERE w2.series_id=s.series_id AND w2.valid_time_utc_ns<=l.lap_start_utc_ns
 ORDER BY w2.valid_time_utc_ns DESC,w2.sample_index DESC LIMIT 1);
CREATE VIEW IF NOT EXISTS v_forecast_weather_at_cutoff AS
 SELECT r.*,v.valid_time_utc_ns,v.lead_time_s,v.air_temperature_c,v.precipitation_probability_pct
 FROM weather_forecast_runs r JOIN weather_forecast_values v USING(forecast_run_id)
 WHERE r.historical_publication_verified=1 AND r.known_available_at_utc IS NOT NULL;
"""

# Prefer observed lap evidence, then lap coverage, then the most recent revision.
# A results-only publication cannot displace laps; a sparse fallback cannot erase
# available sector, tyre, speed and original-clock evidence from a richer version.
_LAP_EVIDENCE_FIELDS = (
    "lap_time_ns", "sector1_ns", "sector2_ns", "sector3_ns",
    "lap_start_clock_ns", "lap_end_clock_ns", "lap_start_utc_ns",
    "sector1_end_clock_ns", "sector2_end_clock_ns", "sector3_end_clock_ns",
    "speed_i1_kmh", "speed_i2_kmh", "speed_finish_line_kmh", "speed_trap_kmh",
    "compound_raw", "tyre_age_laps", "fresh_tyre", "track_status_raw",
    "pit_in_clock_ns", "pit_out_clock_ns",
)
_LAP_EVIDENCE_SCORE = "+".join(f"(l.{name} IS NOT NULL)" for name in _LAP_EVIDENCE_FIELDS)
CURRENT_LAPS_VIEW_SQL = f"""
CREATE VIEW IF NOT EXISTS v_current_laps AS
 WITH ranked AS (
 SELECT v.version_id,row_number() OVER(PARTITION BY v.session_id
 ORDER BY sum({_LAP_EVIDENCE_SCORE}) DESC,count(*) DESC,v.rowid DESC) preference
 FROM session_versions v JOIN laps l ON l.version_id=v.version_id
 WHERE v.status='published' GROUP BY v.version_id,v.session_id,v.rowid)
 SELECT l.* FROM laps l JOIN ranked r USING(version_id) WHERE r.preference=1
"""
SCHEMA_SQL += CURRENT_LAPS_VIEW_SQL + ";"
