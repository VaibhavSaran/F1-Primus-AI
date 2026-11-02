# Project instructions

- Work on the existing `ver_2` branch unless the user directs otherwise.
- Use the project's `primusEnv` for all project Python, package installation,
  tests and tooling. In PowerShell run:
  `& '.\primusEnv\Scripts\python.exe' ...`
  Use `-m pip`, `-m pytest`, `-m streamlit` and other module entry points with
  that executable. Do not silently substitute system Python or another environment.
- Read `.scratch/memory.md` on local resumption when it exists. It is local
  working context; current user instructions take precedence.
- Preserve the user's untracked `test.py`; do not overwrite or commit it.
- Keep temporary plans, environment backups and evidence in ignored `.scratch/`,
  and the persistent knowledge graph in ignored `graphify-out/`. Never force-add
  these directories, virtual environments, caches, local artifacts or secrets.
- Retain Streamlit and the approved integrations. A frontend migration, paid
  service or deployment requires the user's authorization.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the pinned runtime, repair evidence and
rollback instructions.
