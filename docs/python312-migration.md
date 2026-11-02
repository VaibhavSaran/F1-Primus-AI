# Python 3.12.10 migration

The requested target is exact CPython **3.12.10**, Windows x64, at the existing
`.\primusEnv` path. The migration preserves Streamlit 1.55.0,
archives, model weights, notebooks, credentials and the user's `test.py`.
No system Python, default interpreter, registry association or deployment changes
are needed. Python 3.12.10 is the last full maintenance release of 3.12, rather
than the newest 3.12 security patch: the exact requested version is intentional.

## Inventory and dependency decision

The working source environment was Python 3.10.20, with 249 installed
distributions before this task's additive XGBoost install. Application production
requirements had 232 pins; foundation research adds 13 pins and development adds
two. XGBoost adds one, producing 248 application/research/test packages.
Torch additionally requires setuptools on Python >=3.12, producing 249 resolved
target distributions. Setuptools 84.2.1 is pinned conditionally in the foundation
requirements; pip remains a separate bootstrap tool.

The complete per-distribution installed/target/Requires-Python/wheel evidence is
in [the dependency matrix](python312-dependency-matrix.md). A binary-only pip
resolution against Windows cp312 succeeded for the 248 application distributions:
50 platform/native wheels and 198 universal wheels. Actual target installation
additionally selected the universal setuptools wheel because pip cross-target
options do not override every host environment marker. The final matrix uses
the actual target installation report, including all 249 distributions. There is no need to compile
extensions or copy interpreter-specific native files from the old environment.
A freshly downloaded cp310-abi3 wheel, such as protobuf, is compatible with 3.12
through CPython's stable ABI. Wheel tags are checked against the actual target
interpreter rather than rejected merely for containing an earlier Python tag.

Only two production pin decisions accompany the runtime change:

- FastF1 3.8.1 in the old requirements did not describe the working installation,
  which already used 3.9.0. The target pin is 3.9.0, preserving observed behavior.
- XGBoost 3.2.0 is the latest available release supporting both source Python
  3.10 and target 3.12. Versions 3.3/3.4 require Python >=3.12. The constrained
  source install added only XGBoost; existing NumPy/SciPy pins stayed unchanged.

NumPy 2.2.6, pandas 2.3.3, PyArrow 23.0.1, SciPy 1.15.3,
scikit-learn 1.7.2, Torch 2.10.0, TimesFM software 3.0.2, Streamlit 1.55.0,
Anthropic 0.84.0 and all LangChain/LangGraph integration pins remain unchanged.
SDK upgrades require a demonstrated API compatibility need; Python migration
alone does not justify them. TabFM's pinned official source declares Python
>=3.11, so 3.12 removes the old declared-runtime incompatibility. It does not
change the checkpoint or guarantee model quality. TimesFM's software version
remains distinct from the TimesFM 2.5 checkpoint.

## Ordered execution and acceptance gates

1. Capture `pip list`, `pip freeze`, Python version, `pyvenv.cfg`, kernel paths
   and dependency resolution before mutation. Keep ignored evidence under
   `.scratch/`. Never print environment variable values or process command lines.
2. Download exact managed Python 3.12.10 from Astral's pinned
   `python-build-standalone` release into a **new** `.scratch/python312-runtime`
   directory. Preserve the old `.scratch/environment-python` base. The installer
   uses `--no-bin --no-registry`; it does not replace system Python.
3. Create a fresh `.scratch/primusEnv-py312-staging` venv. This staging
   interpreter is used only for authorized migration bootstrap, package installs
   and package smoke checks. Project execution remains on `primusEnv` until
   cutover. Install every package from fresh compatible wheels; do not copy
   site-packages from 3.10. Save the install report and wheel hashes.
4. Run staging `pip check`, numerical/native imports, and inspect all installed
   wheel tags for actual interpreter compatibility. Save the source environment's complete
   working copy before changing its path. The October 23 backup is the original
   broken environment and **is not** the rollback for this migration.
5. Coordinate a quiet cutover with the application owner. Inspect Python
   processes immediately before moving paths. Stop only the exact owned
   Streamlit process after verifying its PID, executable and port. Do not kill
   user notebook kernels: unsaved in-memory notebook state requires the user to
   save and stop the kernel first. Refuse cutover if an unidentified process
   holds the environment open. No collectors or heavy model runs may overlap.
6. Resolve absolute source/target paths under `.` and refuse existing
   destination names. Move working `primusEnv` intact to a unique ignored backup;
   move fresh staging to `primusEnv`. Preserve both base runtime directories.
   Regenerate venv launchers/activation scripts at the final path and reinstall
   the exact lock as needed: Windows console-script launchers embed the staging
   interpreter path, so moving alone is insufficient. All subsequent project
   commands must use `.\primusEnv\Scripts\python.exe`.
7. Check `sys.executable`, `sys.prefix`, `sys.base_prefix`, Python 3.12.10,
   `pip check`, imports, pandas/Arrow roundtrip and small numerical fits. Run the
   full offline suite; inspect unexpected failures rather than hiding skips.
   Verify real archived-data Streamlit pages, prediction and chat workflows;
   execute a bounded actual foundation checkpoint inference when memory allows.
   Old immutable inference receipts remain historical artifacts; new runtime
   results need their own provenance and must not overwrite old evaluations.
8. The existing user `primusenv` kernelspec already points to the retained final
   path. Verify it after cutover; no registration overwrite is needed when it
   still matches. New kernels must use Python 3.12.10. Restart the owned local
   Streamlit app on its original loopback port and verify browser behavior.
9. Update `ENVIRONMENT.md`, CI Python version and Docker base reference to
   3.12.10 after successful cutover. Container building/deployment is a separate
   unperformed action; Linux dependency wheels and the optional foundation set
   need their own container validation if later requested. No Docker runtime is
   needed for this Windows migration. Retain rollback evidence and report any
   unresolved limitation explicitly.

## Local setup commands

These are migration setup commands, an explicit exception to ordinary project
execution remaining in `primusEnv` until cutover:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" python install 3.12.10 `
  --install-dir "$PWD\.scratch\python312-runtime" --no-bin --no-registry `
  --cache-dir "$PWD\.scratch\python312-uv-cache" --no-config
& '.\.scratch\python312-runtime\cpython-3.12.10-windows-x86_64-none\python.exe' `
  -m venv '.scratch\primusEnv-py312-staging'
& '.\.scratch\primusEnv-py312-staging\Scripts\python.exe' -m pip install `
  --only-binary=:all: -r requirements-dev.txt -r requirements-foundation.txt
```

Network-restricted bootstrap attempts may fail; they are not evidence of a
missing upstream package. Network-enabled binary resolution is the compatibility
evidence. Automatic approval review rejected reuse of the old base runtime
directory because it risked rollback integrity; using the new dedicated
directory resolved that restriction without mutating the old runtime.

## Rollback

On native import failure, unresolved package conflict, project regression,
invalid runtime provenance or unrecoverable app startup, stop the verified owned
new server and preserve the failed 3.12 environment under a new `.scratch` name.
Move the complete **working pre-cutover 3.10 backup** back to `primusEnv`.
Because the original final path and old managed base runtime remain intact,
the saved configuration/launchers still reference their original locations.
Run the restored interpreter's version/import checks and `pip check`, then
restart the owned local app. Never delete environments or modify archives,
weights, `.env`, user notebooks, kernel state or `test.py` during rollback.

## Evidence and status

Evidence includes `.scratch/python312-installed-before.json`,
`python312-freeze-before.txt`, `python312-runtime-install.log`,
`python312-resolution.json`, `python312-resolution.log`,
`python312-target-lock.txt`, `python312-stage-install.json` and
`python312-stage-install.log`. The dependency matrix is reviewable source
documentation; full local inventories and downloaded assets remain ignored.

Provisioning, binary installation of all 249 distributions, staged `pip check`,
21 native/library imports, all installed-wheel compatibility checks, Arrow
roundtrip, Ridge/XGBoost tiny fits and a Torch numerical operation succeeded on
3.12.10. Evidence is `.scratch/python312-stage-smoke.json`.
Final cutover succeeded at `primusEnv`; final support audit confirmed exact
3.12.10, all 249 locked distributions plus pip, compatible wheels, regenerated
activation/console scripts, stable kernel path and the working 3.10 backup.
Evidence is `.scratch/python312-cutover.log`,
`.scratch/python312-final-reinstall.json` and
`.scratch/python312-final-support-audit.json`. The parent task records full
project tests and browser acceptance separately. Guarded local procedures are
`.scratch/python312-cutover.ps1` and `.scratch/python312-rollback.ps1`; both
default to dry-run and require an explicit `-Execute` switch for path mutation.

## Primary references

- [CPython 3.12.10 release](https://www.python.org/downloads/release/python-31210/)
- [Python version support status](https://devguide.python.org/versions/)
- [XGBoost installation](https://xgboost.readthedocs.io/en/stable/install.html)
- [XGBoost release metadata](https://pypi.org/project/xgboost/)
- [uv managed Python installation](https://docs.astral.sh/uv/guides/install-python/)
- [Pinned official TabFM source](https://github.com/google-research/tabfm/tree/fbb665569425fd2f490c6576b3af967876fe11ff)
