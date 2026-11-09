# Updating the personal GCP deployment

The custom domain uses the IAP protected `primus-domain-gateway` Cloud Run
service, which forwards to the application on Compute Engine `primus-personal`
in `us-central1-a`. The application reads its serving release from persistent
disk at `/srv/primus/release`. Mutable conversations and tracking records live
at `/srv/primus/state`. Application updates use the existing root owned launcher
at `/srv/primus/run.sh`, which retrieves the Anthropic key from Secret Manager.

## Application changes

Push the reviewed source changes to `main`. The verification workflow runs
first. A successful push verification triggers the deployment workflow, which
builds the application image, resolves its immutable digest and deploys it over
IAP SSH. The launcher checks Streamlit health. A failed launch restores the
previous image. The gateway and its authentication configuration continue to
route to the same VM port.

The existing `GCP_PROJECT_ID` and `GCP_SA_KEY` GitHub secrets supply deployment
credentials. The account behind that key needs Artifact Registry image write
access, IAP tunnel access and the Compute Engine SSH and sudo permissions used
for this VM. Verify those permissions in the first GitHub deployment run.
Interactive CLI access by the project owner does not verify the CI identity.

The deployment workflow accepts successful pushes from this repository's main
branch and manual runs on main. Pull request verification cannot trigger a
privileged deployment. Deployments serialize with a VM file lock. A deployment
briefly replaces the application container, so open sessions may reconnect.

## Data updates

Data collection and publication are operator actions. Run the paced collectors,
then audit the preferred session versions, Parquet hashes, row counts and SQLite
integrity. Build a new serving release with portable catalogue references and
copy its missing objects to a new private Cloud Storage release prefix. Create
SQLite backups with the SQLite backup API while collectors are idle.

Copy the verified release to a separate VM directory, verify its manifests and
test representative sessions before switching `/srv/primus/release`. Restart
the application to clear its cached catalogue and session inventory. Retain the
previous release for rollback. The archive stays on persistent disk and the
reader selects session and driver partitions when a page requests them.

For Kaggle, build a new immutable yearly export from the same audited cohort.
Update the 2026 bundle, coverage and provenance sidecars, README, checksums and
release index together. Publish a version of the existing dataset and verify
the hosted file list and a downloaded copy. Keep the original saved model
cutoffs when adding actual race results for scoring.

For the existing combined Kaggle dataset, the web uploader lists previously
uploaded bundles and retains them in a new version. Replace `f1-2026.zip` and
`publication-index.json` there to transfer only changed files. The CLI version
command uploads every file in its staging folder. Its full-snapshot workflow
can take several hours for this archive.

## November 8 acquisition checkpoint

- Main branch confirmed locally.
- Mexico City and Sao Paulo each have all five sessions published locally.
  Renewed authentication resolved the initial HTTP 401. The 2026 scope has
  100 eligible sessions, with 89 complete and 11 partial publications.
- Mexico independent ERA5 weather has 57 hourly observations from October 30
  through the window ending November 2 at 01:00 UTC. It passed the runner's
  completeness checks.
- Sao Paulo ERA5 remains pending until the collector's November 14 UTC safety
  boundary. Session weather can arrive earlier with FastF1 telemetry.
- The weather Kaggle update is ready with 8,660 rows across 150 event windows.
  The downloaded hosted CSV matches the staged SHA256 and contains Mexico's
  57 rows. F1 Kaggle Version 3 is published with 100 pinned 2026 sessions.
  Its downloaded index preserves all seven years. The 2026 manifest, catalogue
  and a Sao Paulo position sample match local hashes. Catalogue integrity,
  foreign keys, session count and both race publications passed verification.

The hosted application now uses `/srv/primus/releases/20261108-v1` through the
`/srv/primus/release` symlink. The old release remains for rollback. Delta objects
are in `gs://f1-primus-ai-serving-us-central1/deltas/20261108`, and its manifest
identifies the base release and hashes every file. Reconstruct by copying the
base release and overlaying the delta, then verifying hashes and SQLite integrity.
The 717 payload files total 937,215,843 bytes. Upload checksum comparison and
the VM integrity checks passed before activation. The container is healthy.
Browser checks verified Mexico race classification and Sao Paulo classification,
replay playback, weather charts and race-control messages on the custom domain.
