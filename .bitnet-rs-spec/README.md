# .bitnet-rs-spec

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

## Purpose

`.bitnet-rs-spec/` is the durable, repo-owned knowledge base for BitNet-rs spec
rails.

It owns long-lived source-of-truth artifacts such as roadmap entries,
proposals, specs, ADRs, lane trackers, implementation plans, policy references,
support claim maps, and closeouts.

## Namespace boundaries

This namespace **does not** own tool/session state. The following directories are
external-awareness only:

- `.codex/`
- `.spec/`
- `.claude/`
- `.jules/`

Agents may read repo-owned artifacts in `.bitnet-rs-spec/` to decide work, but
must not treat tool/session directories as durable rails.

## Relationship to `docs/` and `policy/`

- `docs/` remains human-facing explanation and contributor guidance.
- `policy/*.toml` remains the live enforcement ledger surface.
- `.bitnet-rs-spec/` may reference these surfaces without duplicating ownership.
