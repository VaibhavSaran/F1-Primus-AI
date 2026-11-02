import sqlite3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def prepared(tmp_path):
    from data.archive import Archive

    archive = Archive(tmp_path / "archive")
    source = archive.source("fixture", "calendar", b'{"extra":null}', "2026-10-23T00:00:00Z")
    archive.save_calendar(
        [{"event_id": "2025-1", "season": 2025, "round_number": 1,
          "event_name": "Test Grand Prix", "circuit_id": "test", "circuit_name": "Test"}],
        [{"session_id": "2025-1-R", "event_id": "2025-1", "session_code": "R"}], source)
    return archive, source


def rows(lap_time=90000000001):
    return {"drivers": [{"driver_id": "driver", "display_name": "Driver"}],
            "teams": [{"team_id": "team", "canonical_name": "Team"}],
            "session_entries": [{"driver_id": "driver", "team_id": "team", "driver_number": "01"}],
            "laps": [{"driver_id": "driver", "lap_number": 1,
                      "lap_time_ns": lap_time, "fresh_tyre": None}]}


def test_revision_is_immutable_and_current_laps_choose_latest(tmp_path):
    archive, source = prepared(tmp_path)
    first = archive.publish_session("2025-1-R", rows(), [], [source], {"normalizer_version": "1"})
    assert archive.publish_session("2025-1-R", rows(), [], [source], {"normalizer_version": "1"}) == first
    second = archive.publish_session("2025-1-R", rows(91000000003), [], [source], {"normalizer_version": "1"})
    assert second != first
    with archive.connect() as conn:
        assert conn.execute("SELECT count(*) FROM session_versions").fetchone()[0] == 2
        assert conn.execute("SELECT lap_time_ns FROM laps WHERE version_id=?", (first,)).fetchone()[0] == 90000000001
        assert conn.execute("SELECT lap_time_ns,fresh_tyre FROM v_current_laps").fetchone()[:] == (91000000003, None)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO laps(version_id,driver_id,lap_number) VALUES(?,?,?)", (first, "unknown", 2))


def test_raw_parquet_retains_unknown_columns_duplicate_time_and_ns(tmp_path):
    archive, source = prepared(tmp_path)
    raw = pd.DataFrame({"Date": pd.to_datetime([1700000000000000001] * 2),
                        "UnknownFutureChannel": [104, 7], "NullField": [None, None]})
    version = archive.publish_session("2025-1-R", rows(),
        [{"kind": "raw_car", "driver_id": "driver", "table": raw}], [source], {})
    with archive.connect() as conn:
        manifest = conn.execute("SELECT * FROM dataset_files WHERE version_id=?", (version,)).fetchone()
    restored = pq.read_table(archive.root / manifest["relative_path"]).to_pandas()
    assert restored["Date"].astype("int64").tolist() == [1700000000000000001] * 2
    assert restored["UnknownFutureChannel"].tolist() == [104, 7]
    assert restored["NullField"].isna().all()
    assert manifest["row_count"] == 2


def test_failed_publish_has_no_sql_manifest_and_retry_recovers_orphan(tmp_path, monkeypatch):
    archive, source = prepared(tmp_path)
    original = archive.insert_rows

    def broken(conn, table, values):
        if table == "laps":
            raise RuntimeError("simulated SQL failure after files finalized")
        return original(conn, table, values)

    monkeypatch.setattr(archive, "insert_rows", broken)
    dataset = [{"kind": "car", "driver_id": "driver", "table": pa.table({"sample_index": [0], "speed_kmh": [104.]})}]
    with pytest.raises(RuntimeError):
        archive.publish_session("2025-1-R", rows(), dataset, [source], {})
    with archive.connect() as conn:
        assert conn.execute("SELECT count(*) FROM session_versions").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM dataset_files").fetchone()[0] == 0
    monkeypatch.setattr(archive, "insert_rows", original)
    archive.publish_session("2025-1-R", rows(), dataset, [source], {})
    with archive.connect() as conn:
        assert conn.execute("SELECT count(*) FROM dataset_files").fetchone()[0] == 1


def test_unknown_normalized_fields_fail_and_source_deduplicates(tmp_path):
    archive, source = prepared(tmp_path)
    assert archive.source("fixture", "calendar", b'{"extra":null}', "2026-10-24T00:00:00Z") == source
    bad = rows()
    bad["laps"][0]["NewProviderField"] = 100
    with pytest.raises(ValueError, match="NewProviderField"):
        archive.publish_session("2025-1-R", bad, [], [source], {})
    with archive.connect() as conn:
        assert conn.execute("SELECT count(*) FROM source_objects").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM session_versions").fetchone()[0] == 0


def test_missing_or_corrupted_published_file_cannot_report_idempotent_success(tmp_path):
    archive, source = prepared(tmp_path)
    datasets = [{"kind": "car", "driver_id": "driver", "table": pa.table({"sample_index": [0]})}]
    version = archive.publish_session("2025-1-R", rows(), datasets, [source], {})
    with archive.connect() as conn:
        path = archive.root / conn.execute("SELECT relative_path FROM dataset_files WHERE version_id=?", (version,)).fetchone()[0]
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="[Cc]orrupt|[Hh]ash"):
        archive.publish_session("2025-1-R", rows(), datasets, [source], {})


def test_backup_is_independently_readable(tmp_path):
    archive, _ = prepared(tmp_path)
    target = tmp_path / "backup.sqlite3"
    archive.backup(target)
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT event_name FROM events").fetchone()[0] == "Test Grand Prix"


def test_weather_versions_keep_duplicate_clocks_and_distinct_forecast_types(tmp_path):
    archive, source = prepared(tmp_path)
    data = rows()
    data["weather_series"] = [{"series_id": "weather", "provider": "fixture", "kind": "track_observation", "source_id": source}]
    data["weather_observations"] = [{"series_id": "weather", "sample_index": i,
        "session_clock_ns": 123, "rainfall_flag": None, "rain_mm": None, "pressure_kind": "station"} for i in range(2)]
    first = archive.publish_session("2025-1-R", data, [], [source], {})
    second = archive.publish_session("2025-1-R", data, [], [source], {"normalizer_version": "2"})
    with archive.connect() as conn:
        assert conn.execute("SELECT count(*) FROM weather_series").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM weather_observations").fetchone()[0] == 4
        series_id = conn.execute("SELECT series_id FROM weather_series WHERE session_version_id=?", (first,)).fetchone()[0]
        assert first != second
        with pytest.raises(sqlite3.IntegrityError, match="forecast"):
            archive.insert_rows(conn, "weather_forecast_runs", [{"forecast_run_id": "bad", "series_id": series_id}])
        archive.insert_rows(conn, "weather_series", [{"series_id": "forecast", "provider": "fixture", "kind": "forecast"}])
        with pytest.raises(sqlite3.IntegrityError, match="forecast"):
            archive.insert_rows(conn, "weather_observations", [{"series_id": "forecast", "sample_index": 0}])


def test_raw_only_revision_and_parser_change_generate_new_versions(tmp_path):
    archive, source = prepared(tmp_path)
    first = archive.publish_session("2025-1-R", rows(), [{"kind": "raw_laps", "table": pa.table({"Extra": [1]})}], [source], {})
    second = archive.publish_session("2025-1-R", rows(), [{"kind": "raw_laps", "table": pa.table({"Extra": [2]})}], [source], {})
    third = archive.publish_session("2025-1-R", rows(), [{"kind": "raw_laps", "table": pa.table({"Extra": [2]})}], [source], {"normalizer_version": "2"})
    assert len({first, second, third}) == 3


def test_partial_parquet_write_does_not_leave_published_files(tmp_path, monkeypatch):
    archive, source = prepared(tmp_path)

    def interrupted(table, where, **kwargs):
        where.write_bytes(b"incomplete")
        raise OSError("disk interruption")

    monkeypatch.setattr(pq, "write_table", interrupted)
    with pytest.raises(OSError, match="interruption"):
        archive.publish_session("2025-1-R", rows(), [{"kind": "car", "table": pa.table({"sample_index": [0]})}], [source], {})
    with archive.connect() as conn:
        assert conn.execute("SELECT count(*) FROM session_versions").fetchone()[0] == 0
    assert list(archive.root.rglob("*.tmp")) == []
    assert list(archive.root.rglob("*.parquet")) == []


def test_pit_visit_view_pairs_clocks_and_exposes_unmatched_events(tmp_path):
    archive, source = prepared(tmp_path)
    data = rows()
    data["laps"] = [{"driver_id": "driver", "lap_number": 1, "pit_out_clock_ns": 2},
        {"driver_id": "driver", "lap_number": 10, "pit_in_clock_ns": 100},
        {"driver_id": "driver", "lap_number": 11, "pit_out_clock_ns": 130},
        {"driver_id": "driver", "lap_number": 20, "pit_in_clock_ns": 200}]
    archive.publish_session("2025-1-R", data, [], [source], {})
    with archive.connect() as conn:
        found = [dict(row) for row in conn.execute("SELECT * FROM v_pit_visits ORDER BY coalesce(pit_in_clock_ns,pit_out_clock_ns)")]
    assert len(found) == 3
    assert found[0]["unmatched_exit"] == 1
    assert found[1]["duration_ns"] == 30
    assert found[1]["entry_lap"] == 10
    assert found[1]["exit_lap"] == 11
    assert found[2]["unmatched_entry"] == 1
    assert found[2]["duration_ns"] is None


def test_idempotent_publish_rejects_corrupt_raw_source(tmp_path):
    archive, source = prepared(tmp_path)
    archive.publish_session("2025-1-R", rows(), [], [source], {})
    with archive.connect() as conn:
        path = archive.root / conn.execute("SELECT raw_path FROM source_objects WHERE source_id=?", (source,)).fetchone()[0]
    path.write_bytes(b"corrupted source")
    with pytest.raises(ValueError, match="[Cc]orrupt"):
        archive.publish_session("2025-1-R", rows(), [], [source], {})


def test_reference_ids_cannot_be_null_and_historical_entry_numbers_remain_versioned(tmp_path):
    archive, source = prepared(tmp_path)
    with archive.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO drivers(driver_id,display_name) VALUES(NULL,'Unknown')")
    first = archive.publish_session("2025-1-R", rows(), [], [source], {})
    revised = rows()
    revised["session_entries"][0].update(driver_number="99", team_name_at_event="Another Team")
    second = archive.publish_session("2025-1-R", revised, [], [source], {})
    with archive.connect() as conn:
        assert conn.execute("SELECT driver_number FROM session_entries WHERE version_id=?", (first,)).fetchone()[0] == "01"
        assert conn.execute("SELECT driver_number FROM session_entries WHERE version_id=?", (second,)).fetchone()[0] == "99"
        archive.insert_rows(conn, "drivers", [{"driver_id": "other"}])
        with pytest.raises(sqlite3.IntegrityError):
            archive.insert_rows(conn, "session_entries", [{"version_id": first, "driver_id": "other", "driver_number": "01"}])


def test_current_laps_prefer_existing_rich_data_over_newer_empty_or_sparse_fallback(tmp_path):
    archive, source = prepared(tmp_path)
    rich = rows()
    rich["laps"][0].update(sector1_ns=30000000000, sector2_ns=30000000000,
        sector3_ns=30000000001, compound_raw="SOFT", tyre_age_laps=1,
        lap_start_clock_ns=123, lap_end_clock_ns=90000000124)
    rich_version = archive.publish_session("2025-1-R", rich, [], [source], {})
    empty = rows()
    empty["laps"] = []
    archive.publish_session("2025-1-R", empty, [], [source], {})
    archive.publish_session("2025-1-R", rows(91000000003), [], [source], {})
    with archive.connect() as conn:
        assert [row[0] for row in conn.execute("SELECT version_id FROM v_current_laps")] == [rich_version]
    revised = {name: [dict(row) for row in values] for name, values in rich.items()}
    revised["laps"][0]["lap_time_ns"] = 90000000002
    latest_rich = archive.publish_session("2025-1-R", revised, [], [source], {})
    with archive.connect() as conn:
        assert [row[0] for row in conn.execute("SELECT version_id FROM v_current_laps")] == [latest_rich]


def test_view_only_migration_retains_existing_tables_and_selects_one_version(tmp_path):
    archive, source = prepared(tmp_path)
    archive.publish_session("2025-1-R", rows(), [], [source], {})
    archive.publish_session("2025-1-R", rows(91000000003), [], [source], {})
    with archive.connect() as conn:
        conn.execute("DROP VIEW v_current_laps")
        conn.execute("CREATE VIEW v_current_laps AS SELECT * FROM laps")
    archive.migrate_views()
    with archive.connect() as conn:
        assert conn.execute("SELECT count(*) FROM laps").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM v_current_laps").fetchone()[0] == 1
        assert conn.execute("SELECT lap_time_ns FROM v_current_laps").fetchone()[0] == 91000000003
