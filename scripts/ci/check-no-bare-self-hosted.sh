#!/usr/bin/env bash
set -euo pipefail

bad=0

echo "Checking for bare self-hosted runner usage..."

if find .github/workflows -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) ! -name 'release.yml' -print0 | xargs -0 rg -n --color=never 'runs-on:[[:space:]]*\[[^]]*self-hosted[^]]*linux[^]]*x64[^]]*\]'; then
  echo "Bare inline self-hosted/linux/x64 runs-on is forbidden." >&2
  bad=1
fi

while IFS=: read -r file line _; do
  window="$(sed -n "${line},$((line+16))p" "$file")"

  if printf '%s\n' "$window" | rg -q '^[[:space:]]*-[[:space:]]*linux[[:space:]]*$' &&
     printf '%s\n' "$window" | rg -q '^[[:space:]]*-[[:space:]]*x64[[:space:]]*$' &&
     ! printf '%s\n' "$window" | rg -q 'group:[[:space:]]*em-ci-' &&
     ! printf '%s\n' "$window" | rg -q '^[[:space:]]*-[[:space:]]*(em-ci|ci-nano|policy-nano|workflow-nano|rust-tiny|rust-medium|rust-large|rust-16gb|cx23|cx33|cx43|cx53|cpx42)[[:space:]]*$'; then
    echo "$file:$line: bare self-hosted block lacks group/capacity labels" >&2
    bad=1
  fi
done < <(find .github/workflows -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) ! -name 'release.yml' -print0 | xargs -0 rg -n --color=never '^[[:space:]]*-[[:space:]]*self-hosted[[:space:]]*$' || true)

exit "$bad"
