# BitNet-rs spec style and namespace doctrine

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

## Durable home

BitNet-rs stores durable source-of-truth rails in:

- `.bitnet-rs-spec/`

This includes the complete chain:

- roadmap
- proposal / PRD
- spec
- ADR
- lane tracker
- implementation plan
- proof expectations
- support and policy references
- closeout memory

## Separation of concerns

- `.bitnet-rs-spec/` owns durable repo knowledge.
- `docs/` explains the system to humans.
- `policy/*.toml` remains live policy enforcement state.

External/tool-specific directories are awareness-only for this lane:

- `.codex/`
- `.spec/`
- `.claude/`
- `.jules/`

Do not treat those directories as durable rails for proposals/specs/ADRs/plans.

## Authoring guidance

When adding or updating durable artifacts:

1. Keep "why" in proposals.
2. Keep required behavior in specs.
3. Keep durable architecture decisions in ADRs.
4. Keep execution sequencing in lane trackers and implementation plans.
5. Keep support-tier claims and live policy in their existing source surfaces,
   referenced from `.bitnet-rs-spec/` when needed.
6. Keep closeouts as durable records of what landed, what proved it, and what
   remains.
