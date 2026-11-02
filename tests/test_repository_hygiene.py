"""Release checks exercise rejection paths without storing working credentials."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repository_hygiene", ROOT / "scripts/check_repository.py")
hygiene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hygiene)


@pytest.mark.parametrize("text,rule", [
    ("api_" + "key=" + repr("unusable" * 5), "credential-literal"),
    ("SERVICE_API_" + "KEY=" + repr("unusable" * 5), "credential-literal"),
    ("token=" + repr("sk-" + "Z" * 32), "provider-key"),
    ("-----BEGIN " + "PRIVATE KEY-----", "private-key"),
    ("login=" + repr("fictional" + "@example.invalid"), "login-email"),
    ("C:" + "/Users/" + "fictional/private", "personal-path"),
])
def test_sensitive_content_detected_without_echoing_values(text, rule):
    assert (1, rule) in hygiene.scan_text(text)


def test_placeholder_is_safe():
    assert not hygiene.scan_text('API_KEY="your-api-key-here"')


@pytest.mark.parametrize("name", [".env.production", ".streamlit/secrets.toml", ".local/state.json", "model.safetensors", "test.py"])
def test_private_candidates_are_rejected_even_if_tracked(tmp_path, name):
    assert hygiene.scan_file(tmp_path / name, name) == [(1, "private-artifact")]


def test_notebook_execution_state_and_attachments_rejected(tmp_path):
    path = tmp_path / "example.ipynb"
    notebook = {"cells": [{"cell_type": "code", "outputs": [{"text": "private"}], "execution_count": 1, "attachments": {"private": {}}}]}
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert {rule for _, rule in hygiene.scan_file(path, path.name)} == {"notebook-execution-state", "notebook-attachments"}
    notebook["cells"][0].update(outputs=[], execution_count=None, attachments={})
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert hygiene.scan_file(path, path.name) == []


def test_ignore_rules_cover_private_inputs():
    paths = ["test.py", ".env", ".env.production", ".streamlit/secrets.toml", ".local/private", ".scratch/private", "graphify-out/private", "primusEnv/private", "model.safetensors", "mlruns/private", "mlflow-artifacts/private", "data/raw/private", "data/processed/private", "data/reports/private"]
    result = subprocess.run(["git", "check-ignore", "--no-index", "--verbose", "--stdin", "-z"], cwd=ROOT, input=("\0".join(paths) + "\0").encode(), capture_output=True, check=True)
    fields = result.stdout.decode().split("\0")[:-1]
    records = list(zip(fields[::4], fields[1::4], fields[2::4], fields[3::4]))
    assert {path for _, _, _, path in records} == set(paths)
    # Machine-local excludes must never be the only protection on a fresh clone.
    assert all(source == ".gitignore" for source, _, _, _ in records), records


def test_docker_excludes_local_work_and_data():
    rules = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".scratch/", "graphify-out/", "mlruns/", "mlflow-artifacts/", "/data/raw/", "/data/processed/", "/data/reports/"} <= rules


def test_release_candidates_pass():
    findings = [(path, line, rule) for path in hygiene.candidate_paths() for line, rule in hygiene.scan_file(ROOT / path, path)]
    assert not findings, findings
