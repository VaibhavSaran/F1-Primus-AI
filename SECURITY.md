# Security and release hygiene

Keep credentials in local environment variables or ignored `.env` files. Never
commit provider keys, F1 account credentials, Streamlit `secrets.toml`, notebook
outputs, private source archives, downloaded checkpoints, runtime databases or
personal scratch scripts. `.env.example` may contain empty values or explicit
placeholders only. The application runs locally; publishing it or its data needs
a separate review of access, provider terms and deployment settings.

Before preparing a GitHub change, run from the repository root in PowerShell:

```powershell
& '.\primusEnv\Scripts\python.exe' scripts/check_repository.py
& '.\primusEnv\Scripts\python.exe' -m pytest tests/test_repository_hygiene.py
git diff --check
git diff --cached --stat
```

The offline checker scans tracked and nonignored untracked candidate files. It
reports paths, line numbers and rule names without printing matched values. It
rejects private artifact paths, likely credential literals, known key formats,
personal home paths, unexpected binaries and executed notebook outputs. Strip
notebook outputs and execution counts before staging; retain any needed evidence
in ignored `.scratch/`. Ignore rules do not protect files already tracked, so the
checker also examines tracked candidates.

These checks are heuristics, not a guarantee that all secrets or personal data
are absent. Review the actual staged diff and use GitHub secret scanning when
available. Do not force-add ignored directories. Docker ignores local archives,
credentials and checkpoints independently of Git ignores.

If a real credential reaches a commit or remote, revoke or rotate it with its
provider immediately. Removing it from the latest file does not remove it from
Git history. Record affected paths and commit IDs privately without copying the
credential. Coordinate any history rewrite with repository collaborators; do
not rewrite shared history automatically. Report suspected exposure privately to
the repository owner, never in a public issue containing sensitive values.
