# BitNet-rs goal routing

This directory is an optional repo-level routing and discovery surface. It may
point an unscoped agent at useful campaign manifests, but it is never a global
work queue, active-lane lock, or executable-state authority.

Campaign-local manifests under `docs/tracking/campaigns/<campaign>/active.toml`
are the executable work authorities. Independent campaigns may advance at the
same time, subject to their declared dependencies, ownership, and shared-surface
collision rules.

## Role

Repo-level routing metadata may own:

- discovery links to one or more campaign manifests;
- human-readable selection hints for otherwise unscoped work;
- links to proposals, specs, ADRs, plans, and status documents.

It does not own executable work-item state, branch ownership, merge policy,
proof commands, product rationale, behavior contracts, durable decisions,
generated metrics, or public support claims.

## Files

```text
.bitnet-rs/goals/active.toml
.bitnet-rs/goals/archive/YYYY-MM-DD-<lane>.toml
```

If `active.toml` is absent or stale, agents use the campaign named by the task,
lane ownership, or explicit scope. Stop only when neither a campaign authority
nor an explicit task scope identifies executable work.

## Relationship to campaign tracking

BitNet-rs also has campaign-local manifests under
`docs/tracking/campaigns/<campaign>/active.toml`. Those campaign manifests
remain authoritative for work-item state, event history, generated dashboards,
branch metadata, proof commands, and campaign-specific merge policy. A
repo-level routing file may link several campaigns; selecting one does not pause
or block the others.

## Manifest shape

```toml
title = "Optional repository routing hints"
owner = "codex-claude"
created = "YYYY-MM-DD"

campaigns = [
  "docs/tracking/campaigns/example/active.toml",
  "docs/tracking/campaigns/another-lane/active.toml",
]

selection_hint = "For unscoped work, inspect these campaigns and choose one ready, collision-free item."
```

## Agent rule

Read `docs/reference/SPEC_SYSTEM.md` before interpreting routing metadata. Then
read the selected campaign manifest, choose exactly one ready work item for the
PR/branch, and run that item's proof commands. This one-item rule is per PR, not
a repository-wide serialization rule.
