# Repo source-of-truth system

BitNet-rs uses a linked source-of-truth stack so humans and agents can find the
right authority without reconstructing intent from chat history, stale status
notes, or generated dashboards.

## Stack

```text
Roadmap
  -> Proposal
    -> Spec
      -> ADR
        -> Implementation plan
          -> Campaign work item
            -> PR
              -> Proof
```

## Artifact roles

| Artifact | Owns | Does not own |
| --- | --- | --- |
| Roadmap | Release direction, milestone framing, lane discovery | Detailed PR order or proof receipts |
| Proposal | Why the work exists, users, alternatives, risks | Behavior contracts or active queues |
| Spec | Required behavior, acceptance examples, proof requirements | Product rationale or PR sequencing |
| ADR | Durable architecture, proof, or operating decisions | Current task lists or generated status |
| Plan | PR order, dependencies, proof commands, rollback | Product strategy or durable decisions |
| Repo goal routing | Optional campaign discovery links and selection hints | Executable work state or a global lane lock |
| Campaign tracker | Campaign-local execution state, events, generated dashboards | Cross-lane product rationale or other campaigns' locks |
| Support tiers | Public claim tier and proof pointer | Feature design |
| Policy ledgers | Exceptions, CI routing, owners, coverage, review dates | Broad architecture |

## BitNet-rs authority mapping

- Proposals live in `docs/proposals/` and answer why a lane exists.
- Specs live in `docs/specs/` and define what must be true.
- ADRs live in `docs/adr/` and record decisions that should still matter after
  the implementation plan is complete.
- Plans live in `plans/<lane>/` and sequence PR-sized work.
- Optional repo routing metadata lives in `.bitnet-rs/goals/`; it may link one
  or more campaigns but is not executable authority. Campaign-local active
  manifests in `docs/tracking/campaigns/<campaign>/active.toml` own campaign
  execution and may advance concurrently.
- Generated campaign dashboards and events remain under
  `docs/tracking/campaigns/<campaign>/generated/` and
  `docs/tracking/campaigns/<campaign>/events/`.
- Public support claims belong in `docs/status/`, hardware matrices, model
  artifact gates, and receipts.
- Policy exceptions and CI routing belong in `policy/*.toml`.
- PR queue disposition rules belong in
  `docs/specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md`,
  `docs/adr/BITNET-ADR-0006-pr-closure-creates-backlog.md`,
  `docs/tracking/PR_QUEUE_DISPOSITION.md`, and
  `policy/pr-dispositions.toml`.
- Generated tracking conflict rules belong in
  `docs/specs/BITNET-SPEC-GENERATED-TRACKING.md` and
  `policy/generated-tracking.toml`.

## Tool and session namespaces

Agent and tool directories are awareness-only unless this file or a linked ADR
explicitly grants them source-of-truth authority. Do not treat these paths as
durable homes for proposals, specs, ADRs, implementation plans, policy ledgers,
or closeouts:

- `.codex/`
- `.spec/`
- `.claude/`
- `.jules/`

They may contain task-local state, external tool inputs, or session artifacts,
but durable BitNet-rs rails stay in the authority mapping above.

## Rules

1. Keep one kind of truth per artifact.
2. Use one semantic artifact per PR unless the selected plan item says otherwise.
3. Specs define behavior; plans define sequencing.
4. Proposals explain why; ADRs record durable decisions.
5. Campaign work items tell agents what to do now and link to the plan/spec/ADR;
   repo goal metadata only helps discover those campaign authorities.
6. Generated status is updated by tools, not by hand; conflicts are resolved by
   repairing source manifests, events, generators, or checkers before
   regenerating.
7. Public claims require support-tier, hardware, model-artifact, or receipt proof.
8. Policy exceptions require owner, reason, coverage, and review date.
9. Closing a PR is a disposition event, not backlog reduction, unless the close
   reason is valid under the PR queue disposition spec and policy ledger.

## Required headers

New proposals, specs, ADRs, and plans should declare the applicable source links
near the top of the file. Use `n/a` where a field does not apply.

```text
Status:
Owner:
Created:
Linked proposal:
Linked specs:
Linked ADRs:
Linked plan:
Linked issues:
Linked PRs:
Support-tier impact:
Policy impact:
```

Existing legacy artifacts may use older headings, but new work should converge
on these fields instead of adding another status vocabulary.

## Agent workflow

Before changing files, agents should:

1. read `AGENTS.md` or `CLAUDE.md`;
2. read this file;
3. select the campaign named by the task or lane ownership; optional
   `.bitnet-rs/goals/active.toml` routing hints may help only when scope is absent;
4. read that campaign's `active.toml`;
5. read the linked plan;
6. read the linked spec for acceptance;
7. read linked ADRs for constraints;
8. inspect `git status --short` for unrelated staged or modified files;
9. pick exactly one ready work item for this PR/branch;
10. implement only that item;
11. run the listed proof commands plus `git diff --check`;
12. update only required status, receipt, policy, or tracking files.

## Stop conditions

Stop and report instead of guessing when:

- no campaign item or explicit task scope exists; absence of repo routing
  metadata alone is not a stop condition;
- a linked proposal, spec, ADR, or plan is missing;
- proof commands cannot run and no substitute evidence is allowed;
- generated status is dirty but no generator command is provided;
- unrelated staged changes exist;
- requested work conflicts with an ADR or claim boundary;
- a public claim lacks support-tier, receipt, hardware, or model-artifact proof.

## Repo routing lifecycle

Optional repo-level routing hints may live at:

```text
.bitnet-rs/goals/active.toml
```

This file may point to several campaign manifests. It does not activate, pause,
serialize, or retire those campaigns. Executable lifecycle state belongs in
each campaign-local manifest and its events.

Archive obsolete routing snapshots under:

```text
.bitnet-rs/goals/archive/YYYY-MM-DD-<lane>.toml
```

Keep at most one optional repo routing file to avoid conflicting discovery
hints. That file is not a global mutex: independent campaign manifests remain
executable concurrently.

## Closeout format

At the end of a lane, write a closeout under `plans/<lane>/closeout.md` with:

- what shipped;
- proof commands and receipts;
- PRs and CI runs;
- generated status, support-tier, and policy updates;
- deferred work;
- claim boundary;
- next lane recommendation.

## Common failure modes

- If a spec becomes a task list, move PR order to `plans/<lane>/implementation-plan.md`.
- If a plan becomes product rationale, move why-text to `docs/proposals/`.
- If campaign work-item state becomes prose, move details to the plan and keep
  the campaign TOML compact.
- If generated status is hand-edited, add or run the generator/checker.
- If support claims drift, add support-tier proof or narrow the claim.
- If policy exceptions become silent debt, add owner, reason, coverage, and review date.
- If a PR grows into multiple semantic changes, split it by artifact or work item.

## What good looks like

A new contributor or agent can answer these questions from repository files
alone:

```text
What are we doing?
Why?
What must be true?
What decision constrains it?
What PR lands next?
What command proves it?
What may we claim?
What must we not claim?
```

If the repo answers those questions without chat history, the source-of-truth
system is working.
