# Linux-equivalent minute budgeting

Linux-equivalent minutes (LEM) are the shared fuel gauge for BitNet-rs CI.
They make hosted Linux, macOS, Windows, Docker, GPU, model, and external review
lanes comparable enough to plan default PR cost honestly.

```text
LEM = wall-clock minutes x runner multiplier
```

LEM is not a license to delete evidence. It is a way to spend verification on
the lanes that return useful proof.

## Operating targets

Ordinary PRs should stay far below the cost of broad full-CI runs:

- preferred default PR budget: `25` LEM or less;
- normal default limit: `35` LEM;
- elevated risk limit: `75` LEM with an explicit reason;
- hard limit: `125` LEM without a budget override;
- default Linux minute rate for planning: `$0.008`.

Those defaults live in `policy/ci-budget.toml`. If the policy file changes,
this document should be updated in the same PR or cite the policy as the source
of truth.

## Runner multipliers

Use the policy multipliers when estimating CI spend. The current standard model
is:

| Runner class | Multiplier |
| --- | ---: |
| Ubuntu/Linux | 1.0 |
| Windows | 2.0 |
| macOS | 10.0 |
| Docker-heavy lane | 6.0 |
| GPU/self-hosted scarce lane | 6.0 |
| External AI review | 1.0 |

The multiplier is an accounting abstraction, not a performance promise. It
captures scarcity, hosted billing weight, setup cost, and opportunity cost.

## Default PR selection rule

Default PR routing follows this order:

1. classify changed files and risk surface;
2. select the cheapest truthful lane;
3. keep required aggregate status meaningful;
4. route deep validation by label, main, schedule, release, or campaign proof.

Expected outcomes:

- docs/control-plane-only changes avoid broad Rust compilation;
- workflow-only changes use workflow validation and safety checks, not docs-light
  routing;
- Rust/build/test changes use the Rust small gate;
- hardware or receipt-only updates use syntax and receipt validation first;
- unknown or mixed changes route to Rust small, not full CI by default.

## Cancellation and timeout policy

Cancellation is a proof decision. Do not cancel useful long-running lanes just
because a newer event appears. Heavy or broad lanes should normally preserve the
started run and allow GitHub's pending-run replacement semantics to collapse
newer queued work.

Short PR validators may cancel on synchronize events when only the newest
commit's result is useful. Label changes are spend/proof decisions; they should
not discard already-started expensive proof unless the workflow explicitly
documents that the lane is safe to cancel.

Timeouts are runaway guards, not budget controls. If a selected lane is too
expensive to let finish, it should not be selected for that event.

## Receipts

Every CI lane should make its spend and result reviewable:

- planned lanes and estimated LEM in `target/ci/ci-plan.json` when available;
- actual runtime and selected/skipped status in lane actuals receipts;
- failure artifacts uploaded when useful;
- large artifacts uploaded on failure or explicit proof lanes rather than every
  default PR;
- optional lanes reported as `skipped-by-policy`, `advisory-failed`, `passed`,
  or `failed`, not silently hidden.

The purpose of LEM is simple: reduce wasted CI so the repo can afford more
verification where it matters.
