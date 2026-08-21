# Tracking Docs Index

## Core Contracts

- [Lane ownership contract](LANE_OWNERSHIP.md)
- [Swarm development roadmap](SWARM_DEVELOPMENT_ROADMAP.md)
- [PR queue disposition](PR_QUEUE_DISPOSITION.md)
- [Generated tracking docs](generated/README.md)

## Lane index (start here)

Every campaign lane and its current state live in the generated dashboards:

- **[Global campaign dashboard](generated/global-dashboard.md)** — one row per
  lane: active item, open PR, state (`ready` / `proposed` / `blocked` /
  `merged`), the next item, and the claim boundary. This is the index of all
  ~48 lanes.
- **[Campaign lane dashboard](generated/lane-dashboard.md)** — lane title,
  current item, and boundary at a glance.

## Find and start your next work item

The `xtask campaign` subcommands are the source-of-truth rails for executing
lane work. They read the authoritative `active.toml` / event logs directly, so
they are always current even if a dashboard is mid-regeneration:

```bash
# List every campaign manifest
cargo run --locked -p xtask --no-default-features -- campaign list

# One lane's full status (items, states, blockers)
cargo run --locked -p xtask --no-default-features -- campaign status <lane>

# The next runnable item for a lane — prints acceptance AND the exact proof
# commands to run for that item (this is your task definition)
cargo run --locked -p xtask --no-default-features -- campaign next <lane>

# Validate a lane manifest + event log before/after editing it
cargo run --locked -p xtask --no-default-features -- campaign check <lane>

# Cross-campaign advisory audit (run before relying on the dashboards)
cargo run --locked -p xtask --no-default-features -- campaign doctor

# Regenerate the dashboards above (do not hand-edit generated files)
cargo run --locked -p xtask --no-default-features -- campaign generate
```

Pick a lane whose state is `ready` in the global dashboard, run
`campaign next <lane>` to get its acceptance criteria and proof commands, and
work that one item on its declared branch. Do not hand-edit generated
dashboards — change the authoritative manifest/`active.toml` and regenerate.

## Notes

- Campaign manifests, campaign-local `active.toml`, and append-only events are authoritative campaign state.
- Generated dashboards under `docs/tracking/generated/` are outputs and should be updated through their generators/checkers.
