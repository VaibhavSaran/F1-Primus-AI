#!/usr/bin/env bash
set -euo pipefail
image=${1:?An immutable application image is required}
if [[ ! "$image" =~ ^us-central1-docker.pkg.dev/f1-primus-ai/f1-primus-ai/f1-primus-ai@sha256:[a-f0-9]{64}$ ]]; then
    echo 'Unexpected deployment image' >&2
    exit 1
fi
exec 9>/srv/primus/deploy.lock
flock -n 9
runner=/srv/primus/run.sh
test -x "$runner"
grep -q '^image=us-central1-docker.pkg.dev/' "$runner"
docker pull "$image"
backup=$(mktemp /srv/primus/run-backup.XXXXXX)
cp -p "$runner" "$backup"
trap 'rm -f "$backup"' EXIT
sed "s|^image=.*|image=$image|" "$backup" > "$runner.next"
chmod 700 "$runner.next"
bash -n "$runner.next"
mv "$runner.next" "$runner"
if ! "$runner"; then
    cp -p "$backup" "$runner"
    "$runner"
    echo 'Deployment failed; previous image restored' >&2
    exit 1
fi
echo 'Deployment health check passed'
