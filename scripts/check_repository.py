"""Offline release hygiene checks; prints locations and rule names, never values.

Scans tracked files and nonignored untracked candidates. Heuristics complement
manual review and GitHub secret scanning; they cannot prove absence of secrets.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider-key": re.compile(r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b"),
    "personal-path": re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b|<|YOUR_|username\b)[A-Za-z0-9_.-]+|/home/(?!<|YOUR_|username\b)[A-Za-z0-9_.-]+|/Users/(?!<|YOUR_|username\b)[A-Za-z0-9_.-]+)", re.I),
    "credential-literal": re.compile(r'''(?i)\b[a-z0-9_]*(?:password|passwd|api_key|access_token|auth_token|client_secret)["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']'''),
    "login-email": re.compile(r'''(?i)\b(?:email|username|login)\s*[:=]\s*["'][^"'\s]+@[^"'\s]+\.[^"'\s]+["']'''),
}
PLACEHOLDERS = ("example", "placeholder", "your_", "your-", "changeme", "test-", "dummy", "fake", "redacted", "os.environ", "getenv", "not-a-real", "replace")
ARTIFACT_SUFFIXES = {".parquet", ".sqlite", ".sqlite3", ".db", ".safetensors", ".pt", ".pth", ".pkl", ".pickle", ".joblib", ".bin", ".ckpt", ".onnx", ".npz", ".npy", ".arrow", ".feather", ".log", ".pem", ".key"}
PRIVATE_DIRS = {".local", ".scratch", "graphify-out", "primusEnv", ".venv", "mlartifacts", "mlruns", ".fastf1_cache"}


def candidate_paths(root: Path = ROOT) -> list[str]:
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root, check=True, capture_output=True)
    return sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})


def scan_text(text: str) -> list[tuple[int, str]]:
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match and not (rule == "credential-literal" and any(word in match.group(1).lower() for word in PLACEHOLDERS)):
                findings.append((number, rule))
    return findings


def scan_file(path: Path, relative: str) -> list[tuple[int, str]]:
    parts = Path(relative).parts
    if (any(part in PRIVATE_DIRS for part in parts)
        or (path.name == "test.py" and len(parts) == 1)
        or (path.name.startswith(".env") and path.name != ".env.example")
        or path.suffix.lower() in ARTIFACT_SUFFIXES or path.name == "secrets.toml"):
        return [(1, "private-artifact")]
    if not path.is_file():
        return []  # A tracked deletion is absent from the proposed tree.
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [(1, "unexpected-binary")]
    findings = scan_text(text)
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(text)
        except ValueError:
            return findings + [(1, "invalid-notebook")]
        if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in notebook.get("cells", [])):
            findings.append((1, "notebook-execution-state"))
        if any(cell.get("attachments") for cell in notebook.get("cells", [])):
            findings.append((1, "notebook-attachments"))
    return findings


def main() -> int:
    paths = candidate_paths()
    findings = [(relative, line, rule) for relative in paths for line, rule in scan_file(ROOT / relative, relative)]
    for relative, line, rule in findings:
        print(f"{relative}:{line}: {rule}")
    present = sum((ROOT / path).is_file() for path in paths)
    print(f"Scanned {present} files from {len(paths)} candidate paths; {len(findings)} findings.")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
