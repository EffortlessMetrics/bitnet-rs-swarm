# Rails framework

`.rails/` is the durable Rails knowledge base for this repository.

`docs/` explains Rails to humans and records adoption guidance.

## Ownership boundaries

Rails owns:

- `.rails/`
- `docs/rails.md`
- `docs/contributing/rails.md`

Rails does not own external, tool-specific state:

- `.codex/` (Codex execution state)
- `.spec/` (Spec Kit / speckit state)
- `.claude/` (external agent/session state)
- `.jules/` (external agent/session state)

Those directories are awareness-only from the Rails framework perspective.
