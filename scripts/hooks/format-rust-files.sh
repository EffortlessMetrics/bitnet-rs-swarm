#!/usr/bin/env bash
set -euo pipefail

rust_files=()
for path in "$@"; do
  if [[ "$path" == *.rs && -f "$path" ]]; then
    rust_files+=("$path")
  fi
done

if ((${#rust_files[@]} == 0)); then
  exit 0
fi

# skip_children keeps formatting scoped to the paths supplied by pre-commit;
# otherwise formatting a module root can rewrite unchanged child modules.
rustfmt --edition 2021 --config skip_children=true -- "${rust_files[@]}"
