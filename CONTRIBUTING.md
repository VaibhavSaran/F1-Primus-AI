# Contributing to Primus AI

## Raise an issue

Search existing issues first. Include the page, race/year/session, reproduction
steps, expected/actual behaviour, runtime versions and a sanitized screenshot or
error. Distinguish missing data from application failures. Proposals should
describe user value and acceptance criteria. Never attach credentials, provider
keys, session tokens, personal paths, private conversations or whole archives.
Follow [SECURITY.md](SECURITY.md) for suspected secret exposure.

## Use a separate branch

Current development uses `ver_2`; `main` is the integration destination. Confirm
the maintainer's current base before branching. In a clean checkout:

```text
git switch ver_2
git switch -c feature/descriptive-change
```

External contributors should fork first. Do not reset shared working branches,
overwrite uncommitted work or force-push over review history. Keep one issue per
branch. Frontend migration, paid services, public access and deployment need
maintainer agreement.

## Implement and verify

Use `primusEnv` and pinned requirements. Add meaningful regression coverage and
update affected docs. Preserve provenance, nullable data, forecast cutoffs, owner
isolation and immutable records. Do not invent telemetry or silently substitute models.

```powershell
& '.\primusEnv\Scripts\python.exe' -m pip check
& '.\primusEnv\Scripts\python.exe' -m pytest tests -q --basetemp=.scratch/pytest-contributor
& '.\primusEnv\Scripts\python.exe' scripts/check_repository.py
git diff --check
```

The normal suite is offline. Live provider tests incur cost and need explicit
operator configuration. Model studies and actual-data checks require local
artifacts. Record commands, dates, denominators and missing prerequisites;
never claim an unrun check passed. Clear notebook outputs/counts before review.

## Open a pull request

Review the exact file list before staging; prefer explicit paths over `git add .`.
Commit only source, sanitized notebooks, tests and appropriate public docs.
Never force-add ignored data, environments, caches, model weights, secrets,
temporary plans or private databases.

Push your feature branch and open a PR against the agreed base (`ver_2` for normal
development; maintainer-approved `ver_2` → `main` for the overhaul). Include:

- Linked issue and concrete before/after behaviour.
- Test commands/results and archive/provider prerequisites.
- Sanitized screenshots for UI changes.
- Data/schema compatibility, limitations and rollback notes.

Address review comments and keep CI green. The maintainer decides when to merge.
Publication and deployment are separate actions; merging code must not upload an
archive or enable paid provider access automatically.
