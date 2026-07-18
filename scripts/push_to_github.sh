#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/push_to_github.sh <git-remote-url>" >&2
  echo "Example: scripts/push_to_github.sh git@github.com:OWNER/KVCacheBench.git" >&2
  exit 2
fi

remote_url="$1"
python scripts/check_release.py

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$remote_url"
else
  git remote add origin "$remote_url"
fi

git push -u origin main
