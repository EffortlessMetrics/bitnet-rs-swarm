# Lunar Lake Active Repository Boundary

Tracking issues:

- https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1036
- https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1125

Created: `2026-05-30`

## Purpose

This document prevents new Lunar Lake work from drifting back into the legacy
public source checkout.

The current Lunar Lake lane is active in:

```text
EffortlessMetrics/bitnet-rs-swarm
```

Routine Lunar Lake campaign, feature, proof, tracker, audit, research, review,
and receipt work belongs in `bitnet-rs-swarm` unless a task explicitly says it
is a public source release, source-to-swarm sync, swarm-to-source promotion, or
external contribution.

## Canonical Active Repo

Use `EffortlessMetrics/bitnet-rs-swarm` for:

- Intel 258V / Lunar Lake campaign work;
- `LNL258V-*` work items and follow-up research issues;
- `bitnet lunar-lake ask`, `validate`, `regress`, `compare`, and receipt work;
- hardware proof, route-policy, regression, comparison, and audit updates;
- docs under `docs/research`, `docs/reviews`, `docs/hardware`,
  `docs/migrations`, and campaign tracking paths;
- small guard, receipt-schema, or docs PRs that support active Lunar Lake work.

The current active Lunar Lake blocker is read from the swarm campaign tracker
and current swarm artifacts. At the time this boundary was written, the active
blocker remained `LNL258V-POWER-006`: real battery-mode low_power telemetry,
route sample evidence, energy-proxy evidence, and benchmark-qualified power
advantage are still missing.

Do not infer active blocker state from old-repo PR text. In particular,
`LNL258V-POWER-013` is historical swarm evidence for AC-only low_power
energy-proxy attempt-versus-valid semantics; it is not the active blocker.
Current low_power promotion remains blocked on `LNL258V-POWER-006` until
battery-mode route samples and energy-proxy evidence exist in swarm.

## Historical Source Repo

Treat `EffortlessMetrics/BitNet-rs` as historical/source-only for Lunar Lake
unless explicitly directed otherwise.

Valid reasons to work in `BitNet-rs` are limited to:

- a public release or publish task;
- a source-to-swarm sync or swarm-to-source promotion packet;
- an explicitly requested public-source hotfix;
- an external contribution or review that must happen in the source repo.

Do not open routine Lunar Lake implementation, audit, tracker, or research PRs
in `BitNet-rs`.

## If A Lunar Lake PR Appears In The Old Repo

Do not merge it as current Lunar Lake evidence.

Use this decision sequence:

1. Confirm whether the PR is a release, sync, promotion, explicit public-source
   hotfix, or external contribution task.
2. If it is not one of those exceptions, close it unmerged.
3. Leave a correction comment that states:
   - active Lunar Lake work lives in `EffortlessMetrics/bitnet-rs-swarm`;
   - the old-repo PR reflects stale or historical source state;
   - no runtime, receipt, tracker, or audit state from that PR should be treated
     as current Lunar Lake evidence.
4. If the idea is still useful, open or link a swarm issue and restate the
   research question, acceptance criteria, and claim boundary there.
5. If any source-repo change already merged by mistake, open a swarm issue to
   classify whether it is historical-only, needs a clean swarm port, or must be
   ignored as stale evidence.

Recommended close comment:

```text
Closing without merge.

The Lunar Lake lane has moved to EffortlessMetrics/bitnet-rs-swarm. This PR was
opened against the legacy EffortlessMetrics/BitNet-rs repo and reflects stale
old-repo tracker state. In swarm, the current Lunar Lake lane keeps
LNL258V-POWER-006 as the active blocker, so this update should not land here.

No runtime, receipt, or tracker state from this PR should be treated as current
Lunar Lake evidence.
```

Recent stale-PR example:

- `EffortlessMetrics/BitNet-rs#6261` was closed unmerged because it targeted
  the legacy source repo, used stale old-repo Lunar Lake numbering, and treated
  `POWER-013` wording as current. The useful idea, if any, is only the guard
  against stale blocker wording; the old-repo PR body, tracker text, CI state,
  and receipt assumptions must not be ported as current Lunar Lake evidence.

## What Can Be Ported

The following may be ported to swarm:

- a research question;
- an issue title;
- a narrow acceptance criterion;
- a claim-boundary guard idea;
- a small docs improvement;
- a test idea, after checking current swarm state.

Port these as new swarm-native issues or PRs. Re-check current swarm `main`
before carrying any text, command, or conclusion forward.

## What Must Not Be Treated As Current Evidence

Do not treat these as current Lunar Lake evidence:

- old-repo PR numbers or tracker statuses;
- stale old-repo `LNL258V-*` numbering;
- old-repo audit wording about active blockers;
- old-repo wording that describes `POWER-013` as active or blocked after swarm
  has returned the active low_power blocker to `LNL258V-POWER-006`;
- source-repo generated dashboards;
- source-repo receipts not present in the current swarm artifact set;
- source-repo CI, bot, or review status;
- closed-unmerged source PR content;
- dense SLM evidence as BitNet QK256/I2_S proof;
- OpenVINO GPU or NPU candidate evidence as route promotion without current
  benchmark-qualified swarm receipts.

If a stale old-repo PR contains a useful idea, port only the idea. Do not port
its old tracker state, receipt state, or completion claim.

## Local Checkout Rule

Use side-by-side clones:

```text
C:\Code\Rust\BitNet-rs
C:\Code\Rust\bitnet-rs-swarm
```

Do not retarget the old source clone by changing its `origin` remote. Start
new active Lunar Lake work from the swarm clone.

Before any Lunar Lake edit, run:

```powershell
git remote -v
git status --short --branch
```

Expected active-work remote:

```text
origin  https://github.com/EffortlessMetrics/bitnet-rs-swarm.git
```

## Claim Boundary

This document is management guidance only.

It does not add:

- new Lunar Lake inference;
- new receipts;
- old-repo PR merge authority;
- route promotion;
- speedup or power-advantage evidence;
- native OpenCL or native NPU proof;
- BitNet QK256/I2_S behavior proof;
- source-to-swarm or swarm-to-source release status.

It only states where active Lunar Lake work belongs and how to handle stale
old-repo PRs.
