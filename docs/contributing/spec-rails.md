# Contributing to repo-native spec rails

Status: active
Owner: repo-architecture
Created: 2026-05-21
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: references-only

## Scope

Use `.bitnet-rs-spec/` for durable repo-owned spec artifacts.

Use `docs/` for explanatory guidance.

Reference `policy/*.toml` for live enforcement ledgers.

## Do not edit as durable rails

The following paths are external/tool-specific state for this lane:

- `.codex/`
- `.spec/`
- `.claude/`
- `.jules/`

These may coexist conceptually, but this system does not migrate, rewrite,
validate, or depend on them as source-of-truth artifacts.

## Practical rules

- Keep the full source-of-truth stack (proposal -> spec -> ADR -> plan -> proof
  -> support/policy -> closeout).
- Keep durable artifacts linked through `.bitnet-rs-spec/index.toml`.
- Keep lane trackers focused and PR-sized; avoid giant shared active queues.
- Keep support-tier and policy claims referenced to their authoritative surfaces.
