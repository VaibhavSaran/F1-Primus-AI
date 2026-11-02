"""SQLite metadata and immutable, atomically published Parquet evidence."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .schema import CURRENT_LAPS_VIEW_SQL, SCHEMA_SQL, SCHEMA_VERSION


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_value(value):
    if isinstance(value, (dict, list, tuple)):
        return canonical(value)
    if value is None or isinstance(value, (str, bytes, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite normalized numeric values must be NULL or explicitly flagged")
        return value
    if hasattr(value, "item"):
        return sql_value(value.item())
    raise ValueError(f"Unsupported normalized value type: {type(value).__name__}")


class Archive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "f1.sqlite3"
        with self.connect() as conn:
            installed = conn.execute("PRAGMA user_version").fetchone()[0]
            if installed not in (0, SCHEMA_VERSION):
                raise ValueError(f"Unsupported archive schema version {installed}")
            # DELETE avoids WAL concurrency bugs in the bundled SQLite 3.50.4.
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.executescript(SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def migrate_views(self):
        """Apply current lap selection to an existing archive without changing tables.

        Call while the collector is stopped; DDL changes commit atomically.
        """
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DROP VIEW IF EXISTS v_current_laps")
            conn.execute(CURRENT_LAPS_VIEW_SQL)

    def _columns(self, conn, table):
        if not re.fullmatch(r"[a-z_]+", table):
            raise ValueError(f"Invalid table name: {table}")
        columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not columns or conn.execute("SELECT type FROM sqlite_master WHERE name=?", (table,)).fetchone()[0] != "table":
            raise ValueError(f"Unknown archive table: {table}")
        return columns

    def insert_rows(self, conn, table, rows):
        columns = self._columns(conn, table)
        known = {r["name"] for r in columns}
        for row in rows:
            unknown = set(row) - known
            if unknown:
                raise ValueError(f"Unknown {table} columns: {', '.join(sorted(unknown))}")
            if not row:
                raise ValueError(f"Empty row for {table}")
            names = list(row)
            quoted = ",".join(f'"{name}"' for name in names)
            values = [sql_value(row[name]) for name in names]
            conn.execute(f'INSERT INTO "{table}" ({quoted}) VALUES ({",".join("?" for _ in names)})', values)

    def _upsert_reference(self, conn, table, row):
        known = {r["name"] for r in self._columns(conn, table)}
        unknown = set(row) - known
        if unknown:
            raise ValueError(f"Unknown {table} columns: {', '.join(sorted(unknown))}")
        primary = [r["name"] for r in self._columns(conn, table) if r["pk"]]
        names = list(row)
        changed = [name for name in names if name not in primary]
        quoted = ",".join(f'"{name}"' for name in names)
        conflict = ",".join(f'"{name}"' for name in primary)
        update = ",".join(f'"{name}"=excluded."{name}"' for name in changed)
        action = f"DO UPDATE SET {update}" if update else "DO NOTHING"
        conn.execute(f'INSERT INTO "{table}" ({quoted}) VALUES ({",".join("?" for _ in names)}) '
                     f'ON CONFLICT({conflict}) {action}', [sql_value(row[name]) for name in names])

    def _safe_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Archive path escapes archive root")
        return path

    def _finalize(self, temporary, final, expected_hash):
        if final.exists():
            if file_hash(final) != expected_hash:
                raise ValueError(f"Corrupt immutable archive object: {final}")
            temporary.unlink()
            return
        # Windows rename refuses replacement; POSIX needs a no-replace hard link.
        try:
            if os.name == "nt":
                os.rename(temporary, final)
            else:
                os.link(temporary, final)
                temporary.unlink()
        except FileExistsError:
            if file_hash(final) != expected_hash:
                raise ValueError(f"Corrupt immutable archive object: {final}")
            temporary.unlink()

    def source(self, provider, request_fingerprint, payload, fetched_at_utc, url=None, **fields):
        if not isinstance(payload, bytes):
            raise TypeError("Raw source payload must be bytes")
        payload_hash = sha256(payload).hexdigest()
        source_id = sha256(canonical([provider, request_fingerprint, payload_hash]).encode()).hexdigest()
        provider_dir = re.sub(r"[^a-zA-Z0-9_.-]", "_", provider)[:64] or "provider"
        relative = f"raw/{provider_dir}/{payload_hash}/payload.bin"
        final = self._safe_path(relative)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f".payload.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._finalize(temporary, final, payload_hash)
        finally:
            temporary.unlink(missing_ok=True)
        if url:
            parts = urlsplit(url)
            secret = re.compile(r"key|token|secret|auth|password|signature", re.I)
            query = [(key, "REDACTED" if secret.search(key) else value) for key, value in parse_qsl(parts.query, keep_blank_values=True)]
            hostname = parts.hostname or ""
            if parts.port:
                hostname += f":{parts.port}"
            url = urlunsplit((parts.scheme, hostname, parts.path, urlencode(query), ""))
        row = dict(source_id=source_id, provider=provider, request_fingerprint=request_fingerprint,
                   first_fetched_at_utc=fetched_at_utc, canonical_url=url,
                   payload_sha256=payload_hash, raw_path=relative, **fields)
        with self.connect() as conn:
            if not conn.execute("SELECT 1 FROM source_objects WHERE source_id=?", (source_id,)).fetchone():
                self.insert_rows(conn, "source_objects", [row])
        return source_id

    def save_calendar(self, events, sessions, source_id):
        with self.connect() as conn:
            source = conn.execute("SELECT first_fetched_at_utc FROM source_objects WHERE source_id=?", (source_id,)).fetchone()
            if source is None:
                raise ValueError("Calendar source is not archived")
            for original in events:
                event = dict(original)
                self._upsert_reference(conn, "seasons", {"year": event["season"], "source_id": source_id,
                    "last_verified_at_utc": source[0], "calendar_status": "available"})
                circuit = {"circuit_id": event.get("circuit_id")}
                for incoming, target in (("circuit_name", "name"), ("timezone_name", "timezone_name"),
                    ("reference_latitude", "reference_latitude"), ("reference_longitude", "reference_longitude"),
                    ("elevation_m", "elevation_m")):
                    if incoming in event:
                        circuit[target] = event.pop(incoming)
                if circuit["circuit_id"] is not None:
                    circuit.update(country=event.get("country"), locality=event.get("location"))
                    self._upsert_reference(conn, "circuits", circuit)
                event.setdefault("calendar_source_id", source_id)
                self._upsert_reference(conn, "events", event)
            for original in sessions:
                session = dict(original)
                session.setdefault("schedule_source_id", source_id)
                self._upsert_reference(conn, "sessions", session)

    def _arrow(self, value):
        if isinstance(value, pa.Table):
            return value.combine_chunks()
        return pa.Table.from_pandas(value, preserve_index=True).combine_chunks()

    def _table_hash(self, table):
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sha256(sink.getvalue()).hexdigest()

    def _verify_files(self, conn, version_id):
        for row in conn.execute("SELECT s.* FROM source_objects s JOIN session_version_sources v "
                                "USING(source_id) WHERE v.version_id=?", (version_id,)):
            path = self._safe_path(row["raw_path"])
            if not path.is_file() or file_hash(path) != row["payload_sha256"]:
                raise ValueError(f"Corrupt or missing published source: {path}")
        for row in conn.execute("SELECT * FROM dataset_files WHERE version_id=?", (version_id,)):
            path = self._safe_path(row["relative_path"])
            if not path.is_file() or file_hash(path) != row["file_sha256"]:
                raise ValueError(f"Corrupt or missing published dataset: {path}")
            if pq.read_metadata(path).num_rows != row["row_count"]:
                raise ValueError(f"Corrupt published dataset row count: {path}")

    def publish_session(self, session_id, tables, datasets, source_ids, metadata):
        tables = {name: [dict(row) for row in values] for name, values in tables.items()}
        metadata = dict(metadata)
        prepared = [{**item, "table": self._arrow(item["table"])} for item in datasets]
        identities = []
        seen = set()
        for item in prepared:
            key = (item["kind"], item.get("driver_id"))
            if key in seen:
                raise ValueError(f"Duplicate dataset partition: {key}")
            seen.add(key)
            identities.append({"kind": item["kind"], "driver_id": item.get("driver_id"),
                               "content_hash": self._table_hash(item["table"])})
        fingerprint = {"session_id": session_id, "tables": tables, "datasets": identities,
                       "sources": source_ids, "metadata": {key: value for key, value in metadata.items()
                       if key not in ("ingestion_run_id", "created_at_utc")}, "schema_version": SCHEMA_VERSION}
        dataset_hash = sha256(canonical(fingerprint).encode()).hexdigest()
        version_id = sha256(f"{session_id}:{dataset_hash}".encode()).hexdigest()
        with self.connect() as conn:
            existing = conn.execute("SELECT status FROM session_versions WHERE version_id=?", (version_id,)).fetchone()
            if existing:
                if existing[0] != "published":
                    raise ValueError("Existing version is not published")
                self._verify_files(conn, version_id)
                return version_id
            if not conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone():
                raise ValueError(f"Unknown session: {session_id}")
        now = utc_now()
        files = []
        for index, item in enumerate(prepared):
            table = item["table"]
            if not item["kind"].startswith("raw_"):
                for name, value in (("version_id", version_id), ("driver_id", item.get("driver_id"))):
                    if name not in table.column_names:
                        table = table.append_column(name, pa.array([value] * table.num_rows, type=pa.string()))
            directory = self._safe_path(f"parquet/{version_id}")
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / f".{uuid4().hex}.parquet.tmp"
            try:
                pq.write_table(table, temporary, compression="zstd", version="2.6")
                with temporary.open("rb+") as stream:
                    os.fsync(stream.fileno())
                if pq.read_metadata(temporary).num_rows != table.num_rows:
                    raise ValueError("Parquet validation row count mismatch")
                digest = file_hash(temporary)
                relative = f"parquet/{version_id}/{index:03d}-{digest}.parquet"
                self._finalize(temporary, self._safe_path(relative), digest)
            finally:
                temporary.unlink(missing_ok=True)
            record = dict(file_id=sha256(relative.encode()).hexdigest(), version_id=version_id,
                source_id=item.get("source_id"), kind=item["kind"], driver_id=item.get("driver_id"),
                relative_path=relative, file_sha256=digest, schema_version=str(SCHEMA_VERSION),
                row_count=table.num_rows, compression="zstd", transform_version=item.get("transform_version"),
                published_at_utc=now)
            for column, first, last in (("session_clock_ns", "first_clock_ns", "last_clock_ns"),
                                         ("timestamp_utc_ns", "first_utc_ns", "last_utc_ns")):
                if column in table.column_names and table.num_rows:
                    bounds = pc.min_max(table[column]).as_py()
                    record[first], record[last] = bounds["min"], bounds["max"]
            files.append(record)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            # A second process may have published while files were being finalized.
            if conn.execute("SELECT 1 FROM session_versions WHERE version_id=?", (version_id,)).fetchone():
                self._verify_files(conn, version_id)
                return version_id
            version = dict(version_id=version_id, session_id=session_id,
                ingestion_run_id=metadata.get("ingestion_run_id"), created_at_utc=now,
                normalizer_version=metadata.get("normalizer_version", "1"), dataset_hash=dataset_hash,
                status="staged", t0_utc_ns=metadata.get("t0_utc_ns"),
                observed_start_utc_ns=metadata.get("observed_start_utc_ns"),
                observed_end_utc_ns=metadata.get("observed_end_utc_ns"),
                completeness_json=metadata.get("completeness_json", metadata.get("completeness")),
                quality_flags_json=metadata.get("quality_flags_json", metadata.get("quality_flags")),
                metadata_json=metadata)
            self.insert_rows(conn, "session_versions", [version])
            for source in source_ids:
                reference = {"source_id": source, "dataset_kind": "session"} if isinstance(source, str) else dict(source)
                self.insert_rows(conn, "session_version_sources", [{"version_id": version_id, **reference}])
            for name in ("drivers", "teams", "circuit_layouts"):
                for row in tables.pop(name, []):
                    self._upsert_reference(conn, name, row)
            # Session-scoped weather identity must change when the containing version changes.
            series_ids = {row["series_id"]: sha256(f'{version_id}:{row["series_id"]}'.encode()).hexdigest()
                          for row in tables.get("weather_series", [])}
            ordered = ["session_entries", "session_results", "laps", "track_status_events",
                       "session_status_events", "race_control_messages", "circuit_markers",
                       "pit_stop_records", "weather_series", "weather_observations", "coverage"]
            unknown_tables = set(tables) - set(ordered)
            if unknown_tables:
                raise ValueError(f"Unsupported session tables: {', '.join(sorted(unknown_tables))}")
            for name in ordered:
                normalized = []
                for original in tables.get(name, []):
                    row = dict(original)
                    if name == "weather_series":
                        row["series_id"] = series_ids[row["series_id"]]
                        row["session_version_id"] = version_id
                    elif name == "weather_observations":
                        row["series_id"] = series_ids.get(row["series_id"], row["series_id"])
                    else:
                        row["version_id"] = version_id
                    normalized.append(row)
                if normalized:
                    self.insert_rows(conn, name, normalized)
            if metadata.get("coverage") and not tables.get("coverage"):
                self.insert_rows(conn, "coverage", [{"version_id": version_id, "checked_at_utc": now, **row}
                                                      for row in metadata["coverage"]])
            self.insert_rows(conn, "dataset_files", files)
            self._verify_files(conn, version_id)
            conn.execute("UPDATE session_versions SET status='published' WHERE version_id=?", (version_id,))
        return version_id

    def backup(self, target):
        target = Path(target).resolve()
        if target == self.db_path or target.exists():
            raise ValueError("Backup destination must be a new file")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source:
            with sqlite3.connect(target) as destination:
                source.backup(destination)
