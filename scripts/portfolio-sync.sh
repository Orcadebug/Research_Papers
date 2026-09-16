#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -z "$(git status --porcelain)" ]]; then
  echo "Portfolio is already synchronized locally."
  exit 0
fi

git add -A

while IFS= read -r path; do
  case "$path" in
    README.md|.gitignore|scripts/*|*/README.md|*/src/*|*/tests/*|*/experiments/*|*/benchmarks/*|*/report/*|*/assets/*|*/docs/*)
      ;;
    *)
      echo "Refusing to publish non-portfolio path: $path" >&2
      git restore --staged -- "$path"
      exit 1
      ;;
  esac
done < <(git diff --cached --name-only)

git diff --cached --check
git commit -m "${1:-chore: synchronize portfolio}"
git push origin HEAD
echo "Published $(git rev-parse --short HEAD) to $(git remote get-url origin)"
