# Contributing to Rails artifacts

Use `.rails/` as the durable source-of-truth framework for this repository.

## Artifact responsibilities

- proposals = why / PRD / alternatives / success criteria
- specs = behavior contracts and evidence requirements
- ADRs = durable architecture decisions
- lanes = focused implementation trackers
- implementation plans = PR-sized execution sequence
- support = product claim to proof mapping (or references to support docs)
- policy = references to live enforcement ledgers
- closeouts = what landed, what proved it, what remains

## Rules

- Keep durable Rails artifacts under `.rails/`.
- Do not migrate or modify `.spec/` as part of Rails ownership.
- Do not create one giant shared active queue.
- Use focused lane trackers under `.rails/lanes/`.
- Every owned artifact must link through `.rails/index.toml`.
- No owned artifact path may live under `.codex/`, `.spec/`, `.claude/`, or `.jules/`.
