# Heavy Rust Lane Consolidation & Cache — Design

Status: **proposal** (design-only; no workflow changes in this PR)
Audience: CI maintainers / fleet admins
Related: `docs/ci/cost-and-verification-policy.md`, `docs/ci/pr-gate-success.md`,
`.github/workflows/em-ci-routed-rust.yml`, `.github/workflows/ci-core.yml`

## Problem

The self-hosted fleet is **saturated** (1× CX53, a couple CX43, lots of
CX33/CX23; more capacity planned but not yet online). Two cost drivers
dominate:

1. **Cold compiles.** `em-ci-routed-rust` builds inside a container with
   `sccache` + a persistent 60 GB `/mnt/ci-cache`, but the other heavy Rust
   lanes (`ci-core`, `compatibility`, `crossval`, `validation`,
   `feature-matrix`, `security`, `test-framework`, `model-gates`, …) use only
   `Swatinem/rust-cache` (network-bound GitHub Actions cache, ~10 GB cap) and
   **no compiler cache**. They rebuild ~200 crates from scratch on the
   scarcest boxes, every run.
2. **Fan-out.** ~25 heavy workflows trigger per PR. `pr-gate` already only
   *waits for* the lanes relevant to a PR's changed files (via `xtask ci
   plan`), but the lane **workflows still run** even when the gate ignores
   them.

## What already exists (build on it, don't reinvent)

- **`pr-gate` is the intended single required check.** It consumes
  `xtask ci plan`, builds a dynamic `required_jobs` list from the PR's changed
  files, and polls the Checks API for those conclusions. It explicitly does
  *not* add lanes that the change set doesn't touch.
- **`ci-core` is the reference scoping pattern.** It has:
  - broad-but-bounded `paths:` triggers (Rust + docs/policy/receipt surfaces);
  - a cheap `classify-changes` job computing `no_rust_inputs` / `docs_only` /
    `policy_docs_only` / etc.;
  - heavy jobs gated on `if: needs.classify-changes.outputs.* != 'true'`;
  - a **Success aggregator that reports truthfully even when compilation is
    skipped** — so the `CI Core Success` check always reports a conclusion.
- **`em-ci-routed-rust` is the cache-warm executor** (container + sccache +
  `/mnt/ci-cache`, org-discovery routing to idle cx43/cx53, normalized
  "BitNet Rust Small Result").
- **Routing is done** (PRs #1231 + #1237 merged): control-plane →
  `em-ci-nano` (CX23), light Rust → `em-ci-tiny` (CX33), heavy → `em-ci-small`
  with `rust-medium` / `rust-heavy-medium`.
- **Shared-cache plumbing exists** (PR #1247): `.github/actions/rust-shared-cache`
  (fail-open `sccache` setup, persistent `/mnt/ci-cache` when present).

## Hard constraint: branch protection requires individual checks (confirmed Case B)

The agent has no read access to `main`'s branch-protection config, but a
**merge attempt on #1237 confirmed the regime empirically**: the API rejected
the merge with

```
405: Required status check "BitNet Rust Small Result" is expected.
```

So `main` requires **individual named status checks** (at minimum the
routed-rust normalized `BitNet Rust Small Result`), with "require branches up to
date before merging" on — it is **not** `PR Gate`-only. This is the conservative
"Case B" the design already assumed, now confirmed, with two consequences:

1. Consolidation (Phase C) **cannot** simply drop retired lanes — each required
   check name must keep reporting (truthful Success or wrapper) until the
   required-checks list is edited in lockstep (admin action).
2. Merging agent PRs needs the branch **up to date**; on the saturated fleet,
   prefer **auto-merge** (GitHub drives update→green→merge) over manual
   refresh-then-merge, which races main and re-cancels in-flight runs.

GitHub's behavior is asymmetric and must be respected:

- A workflow **skipped by `paths:`/`branches:` filters** leaves a *required*
  check **Pending** → wedges the PR.
- A job **skipped by an `if:` condition reports success** → safe for required
  checks.

**Rule for this design:** never let a required check go missing. Either keep the
workflow triggering and emit a truthful Success (the `ci-core` pattern), or
preserve the check name with a thin always-reporting wrapper. The full
required-checks set should still be enumerated (admin) before Phase C so every
name is accounted for.

## Design — three phases, cheapest/safest first

### Phase A — Shared compiler cache (safe now; biggest single win)

Roll `.github/actions/rust-shared-cache` out to every heavy host-run Rust lane
(the `ci-core` canary in #1247 proves it first). No topology change, no
check-name change, fail-open.

- Order by cost: `crossval`, `compatibility`, `ci-core` (done), `validation`,
  `feature-matrix`, `security`, `test-framework`, `model-gates`,
  `quant-matrix`, `property-tests`, `gguf_build_and_validate`, …
- Keep `Swatinem` for registry/target; `sccache` handles compiler output.
- Acceptance: second run on `em-ci-small` shows `sccache --show-stats` cache
  hits and a materially lower wall-clock.

### Phase B — Per-lane change scoping (safe; mirrors `ci-core` + `xtask ci plan`)

Bring each heavy lane up to the `ci-core` pattern so irrelevant PRs skip the
*work* while the *check still reports*:

1. A cheap `classify-changes` (or shared composite) job on `em-ci-nano`.
2. Heavy jobs gated with `if:` on the relevant classification.
3. A Success job that always reports a truthful conclusion.

Relevance per lane **must match `xtask ci plan`** (the gate's source of truth)
so `pr-gate` never waits on a lane that skipped its work. Where a lane already
has bespoke `paths:`, keep them; add the `if:`-gate + truthful Success only
where missing. Prefer extending `xtask ci plan` to emit a per-lane relevance
matrix that both `pr-gate` and the lanes consume, eliminating drift.

### Phase C — Consolidate behind `em-ci-routed-rust` (needs branch-protection sign-off)

Fold the normal heavy lanes into the routed executor as **matrix shards** that
share the warm `sccache`/`/mnt/ci-cache`, so a couple CX43 + one CX53 aren't
fought over by 25 independent workflows:

- One routed workflow, matrix = { ci-core, compatibility, crossval-shards,
  validation, … }, each shard `em-ci-small` + `rust-medium`
  (`rust-heavy-medium`/`rust-large` reserved for the genuine heaviest).
- Emit normalized result check(s) (extend the existing "BitNet Rust ... Result").
- **Check-name preservation:** until branch protection is confirmed
  `PR-Gate-only`, keep each retired lane's required check name alive with a
  thin wrapper job that mirrors the shard's conclusion. Renaming/removing a
  required check before updating branch protection wedges merges.

## Runner tiering (route to the smallest safe tier by actual workload)

Hosted Linux is **exception-only**; self-hosted `em-ci-*` is the default.
Cross-repo queueing on the shared groups is acceptable backpressure — do **not**
add `ubuntu-latest` fallback just to avoid a queue.

| Workload | Group | Key labels |
|----------|-------|------------|
| non-build / control / policy / workflow metadata | `em-ci-nano` | `workflow-nano` / `policy-nano` |
| review / LLM / droid review | `em-ci-review` | `review-nano` / `llm-review` / `droid-review` |
| tiny Rust (`cargo metadata`, small `cargo check`, fmt, light xtask) | `em-ci-tiny` | `rust`, `rust-tiny` / `backfill-rust-small` |
| normal Rust builds / test shards / clippy / docs | `em-ci-small` | `rust`, `rust-small` / `rust-medium` |
| heavy Rust (coverage, full compatibility, big linking, model gates, large matrix) | `em-ci-small` | `rust`, `rust-heavy-medium` / `rust-16gb` |
| single heaviest lane only | `em-ci-small` | `rust`, `rust-large` |

All self-hosted lanes carry `[self-hosted, Linux, X64, em-ci, trusted-pr]`.
`rust-large` is reserved for the one or two genuinely heaviest jobs — if every
repo marks its hardest job `rust-large`, the whole org serializes behind the one
CX53.

### Hosted-runner exceptions (the only allowed cases)

macOS/iOS signing; Windows-specific tests; untrusted **fork** PR code; a one-off
migration bootstrap (file an issue); a job needing an unavailable
OS/tool/hardware; or explicit owner approval. Everything else queues on
`em-ci-*` (a self-hosted job may sit queued up to GitHub's 24 h limit — that is
acceptable, paid hosted capacity is not the escape hatch).

## Concurrency policy (classify before changing — never mass-flip)

- **proof / evidence / gate-polling / nightly / perf** → `cancel-in-progress:
  false` so the in-progress proof finishes and only the latest *pending* run
  waits. The current `false` settings on these lanes are intentional; do not
  blindly flip them.
- **cheap lint / format / metadata PR checks** → PR-only cancellation is fine
  where repo policy allows (this is why `ci-core`'s
  `cancel-in-progress: ${{ github.event_name == 'pull_request' }}` is correct
  and stays as-is).

Always use a per-workflow/per-ref group so stale pending runs don't pile up.

## Safety guards (required before expanding self-hosted usage)

A *queued* job is fine; a *hung / disk-full / wedged* runner is not. Each
migrated lane relies on (platform action or repo fallback): `trusted-pr` gating,
explicit group+labels, a disk preflight (`>=20 GB` free), `if: always()`
workspace cleanup, a bounded `SCCACHE_CACHE_SIZE`, a `timeout-minutes` on every
job, no privileged Docker / host socket without approval, and no fork-PR secrets
on self-hosted. Avoid casual `docker system prune -af --volumes` in repo lanes —
that is runner-tier policy.

## Branch-protection change (human / admin gate)

This is the only step the agent cannot self-serve. One of:

- **Case A — confirm `PR Gate` is the only required check.** Then Phase C can
  drop wrappers and `pr-gate` decides per PR. Best end state.
- **Case B — individual lanes are required.** Keep wrapper checks, or update
  the required-checks list to `PR Gate` only in lockstep with Phase C.

Until confirmed, treat as Case B (conservative): keep every current check
reporting.

## Sequencing

1. Land routing guardrails (#1237) and pre-commit shift-left (#1240).
2. Prove the cache canary (#1247) → **Phase A** rollout.
3. **Phase B** scoping, lane by lane, aligned to `xtask ci plan`.
4. Confirm branch protection → **Phase C** consolidation.
5. As new CX boxes land, add them as canary labels first, not directly into the
   `rust-medium` pool.

## Risks & rollback

| Risk | Mitigation |
|------|------------|
| `pr-gate` waits on a skipped lane | Phase B relevance must equal `xtask ci plan`; prefer a single shared relevance matrix |
| Required check disappears | Always emit truthful Success / keep wrapper until branch protection confirmed |
| `/mnt/ci-cache` disk fill | `SCCACHE_CACHE_SIZE` cap + reuse routed-rust disk-guard; per-runner dirs |
| sccache binary/runner drift | action is fail-open (builds uncached on miss) |
| Each phase | Independently revertable; no phase deletes a check before protection is updated |

## Migration PR acceptance checklist

A heavy-lane migration/consolidation PR is acceptable when it answers:

- [ ] Which branch-protection checks are required (or: assumed individual)?
- [ ] Which jobs still use hosted Linux, and which exception applies?
- [ ] Are required check names preserved (truthful Success / wrapper)?
- [ ] Are `paths:` filters applied only where the check still reports?
- [ ] Is each job on the smallest safe tier per the table above?
- [ ] Do heavy Rust jobs use shared `sccache`/cargo cache or the routed path?
- [ ] Are proof/evidence concurrency semantics preserved (no blind flips)?
- [ ] Are disk guard, cleanup, cache bound, and timeouts present?
- [ ] Are fork/untrusted PRs kept off trusted self-hosted runners?
- [ ] Is `rust-large` reserved for only the heaviest lane(s)?

## Non-goals

Release / GPU / intel-gpu / rocm / macOS / Apple / ci-image / repin workflows
are out of scope and remain on their current runners.
