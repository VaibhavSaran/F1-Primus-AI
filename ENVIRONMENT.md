# Python environment

Run all project Python through `primusEnv`, including installs and tests:

```powershell
& '.\primusEnv\Scripts\python.exe' --version
& '.\primusEnv\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\primusEnv\Scripts\python.exe' -m pip check
& '.\primusEnv\Scripts\python.exe' -m pytest tests -q
& '.\primusEnv\Scripts\python.exe' -m streamlit run streamlit_app/app.py
```

Production package versions remain pinned in `requirements.txt`. Development
adds `pytest==8.4.2` and its required `iniconfig==2.3.1`; the other pytest
dependencies already have production pins. Use exact Python **3.12.10**, Windows
x64. The working environment was rebuilt from fresh compatible wheels on
November 1, 2026; no interpreter-specific binaries were copied from Python 3.10.
Optional foundation research dependencies remain in `requirements-foundation.txt`.

## Current runtime and migration evidence

The managed base runtime is
`.scratch\python312-runtime\cpython-3.12.10-windows-x86_64-none`.
Keep it: `primusEnv\pyvenv.cfg` references this directory. It was installed with
Astral uv using `--no-bin --no-registry`, without changing system Python.

For a new checkout, provision the base and create the final environment:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" python install 3.12.10 `
  --install-dir "$PWD\.scratch\python312-runtime" --no-bin --no-registry `
  --cache-dir "$PWD\.scratch\python312-uv-cache" --no-config
& '.\.scratch\python312-runtime\cpython-3.12.10-windows-x86_64-none\python.exe' `
  -m venv primusEnv
& '.\primusEnv\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\primusEnv\Scripts\python.exe' -m pip install -c requirements.txt -r requirements-foundation.txt
```

Never recreate an existing environment without preserving it first. The detailed
procedure is [Python migration](docs/python312-migration.md), with the complete
[249-package wheel matrix](docs/python312-dependency-matrix.md). FastF1 is pinned
to the already installed working 3.9.0 version; XGBoost 3.2.0 is additive.
Torch's Python >=3.12 setuptools dependency is explicitly pinned conditionally.
Streamlit remains 1.55.0.

Final cutover completed at the same `primusEnv` path. Evidence under `.scratch/`:

- `python312-cutover.log` and `python312-final-reinstall.json`: successful fresh
  binary reinstall at the final path, including regenerated console launchers.
- `python312-stage-smoke.json`: native imports and numerical checks passed.
- `python312-final-support-audit.json`: Python/base paths, all 249 exact locked
  versions, installed-wheel compatibility, stable kernel path and absence of
  staging references passed. Pip makes 250 installed distributions in total.
- `python312-working310-manifest.json`: working pre-cutover environment inventory.

The existing `primusenv` Jupyter kernelspec still points to
`.\primusEnv\Scripts\python.exe`. Activation scripts were regenerated
at the final path; `pip.exe`, `streamlit.exe` and `pip check` passed directly.
The parent task records full project-test and browser acceptance separately;
this runtime support verification alone is not a full application test.

## Current migration rollback

The complete **working** Python 3.10.20 environment is preserved at
`.\.scratch\environment-backup-20261101-py310\primusEnv`.
Keep its original base at
`.scratch\environment-python\cpython-3.10.20-windows-x86_64-none` too.
After saving/stopping notebook kernels and stopping the verified owned server,
use the guarded `.scratch\python312-rollback.ps1` procedure. It defaults to dry
run; `-Execute` preserves the failed target under a new ignored path and moves
the complete working backup back to `primusEnv`, then checks Python and packages.
It refuses active project Python processes and existing preservation targets.
Do not delete environments, archives, weights, user notebooks, `.env` or `test.py`.

## Historical October 23 repair and evidence

The following records the earlier 3.10 repair; it is historical evidence rather
than the current runtime setup procedure.

The original environment declared Python 3.10.0 but launched `C:\Python` 3.12.10.
Its cp310 native extensions failed to import. Task 0 restores the interpreter
alignment while retaining the installed production packages.

The base CPython 3.10.20 runtime is installed locally at
`.scratch\environment-python\cpython-3.10.20-windows-x86_64-none` by Astral uv's
managed Python installer, from its upstream `python-build-standalone` release.
Keep that directory: the working 3.10 rollback environment references it. This base runtime is
setup support, not a second application environment. Nothing is installed into
system Python or registered as a system default.

The exact bootstrap command is:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" python install 3.10.20 `
  --install-dir "$PWD\.scratch\environment-python" --no-bin --no-registry `
  --cache-dir "$PWD\.scratch\environment-uv-cache" --no-config
```

For a new checkout, install that base runtime first, create `primusEnv` with
its `python.exe -m venv primusEnv`, then use only the `primusEnv` commands above.
Do not recreate an existing environment without preserving it first.

Local evidence and full package inventories are intentionally ignored:

- `.scratch/environment-before.json` and `environment-before-freeze.txt`:
  interpreter, native-extension and package metadata before repair.
- `.scratch/environment-backup-manifest.json`: all 49,629 files matched by size
  and timestamp, plus 754 SHA-256 comparisons of native binaries, launchers,
  configuration and package metadata, before any environment mutation.
- `.scratch/environment-imports-after.json`: actual core-import and numerical
  smoke-check results.
- `.scratch/environment-after.json` and `environment-after-freeze.txt`:
  final runtime and resolved package inventory.
- `.scratch/m0-runtime-log.md`: commands, outcomes and any limitations.

## Historical original broken-state backup

The original complete environment is retained at
`.\.scratch\environment-backup-20261023\primusEnv`.
It restores the **original broken state**, useful for recovering original files
or comparing changes; it is not a working alternative runtime.

Use the November 1 working backup for migration rollback. This October backup
is for original-file recovery and historical comparison only.
The full backup and base runtime are local files and must not be committed or
included in container build contexts.
