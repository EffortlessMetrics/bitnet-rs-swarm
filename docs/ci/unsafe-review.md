# unsafe-review policy

`unsafe-review` is advisory unsafe-contract review. It checks whether changed
unsafe seams have reviewable evidence: a safety contract, local guard, test
reach, and witness route.

It does not prove memory safety or UB-free status unless a matching execution
witness, such as a Miri or sanitizer receipt, is attached.

## Tool boundary

| Tool | Question answered |
| --- | --- |
| `cargo-allow` | Is this unsafe or source exception allowed, owned, and visible? |
| `unsafe-review` | Is this unsafe seam reviewable: contract, guard, test reach, and witness route? |
| Miri/sanitizers | Did a concrete execution expose UB or memory misuse? |

These tools are complementary. A `cargo-allow` entry can authorize retaining an
unsafe block, but it does not make the unsafe contract reviewable. A clean
`unsafe-review` card can show that the seam is documented and witnessed, but it
is not a proof of all possible executions.

## When to run

Run `unsafe-review` for PRs that change or expose:

- `unsafe` blocks or `unsafe fn` bodies;
- FFI, C ABI, raw pointer, layout, alignment, or aliasing boundaries;
- native, GPU, or SIMD kernels;
- parser/tokenizer or binary-format code with unchecked indexing or layout
  assumptions;
- suppressions or policy entries that affect unsafe reviewability.

Ordinary docs or control-plane PRs do not need an unsafe-review lane unless they
change the policy for unsafe evidence.

## Expected artifacts

The standard artifact directory is:

```text
target/unsafe-review/
  cards.json
  pr-summary.md
  github-summary.md
  cards.sarif
  comment-plan.json
  witness-plan.md
  lsp.json
  receipt-audit.json
```

The PR summary should distinguish these outcomes:

- `passed`: reviewed seams have contracts and expected witnesses;
- `advisory-failed`: review gaps exist, but the lane is not yet blocking;
- `skipped-by-policy`: no unsafe-relevant surface changed;
- `unavailable`: the tool or witness runner was unavailable.

Skipped or unavailable review must not be presented as a memory-safety proof.

## Policy files

Unsafe-review policy and suppressions live under `policy/`:

```text
policy/unsafe-review.toml
policy/unsafe-review-suppressions.toml
policy/unsafe-witnesses.toml
```

Suppressions must include an owner, reason, coverage or witness plan, creation
date, review date, and expiry when temporary. Broad suppressions require a
separate explanation of why the selector cannot be narrower.

## Review questions

For each changed unsafe seam, reviewers should be able to answer:

1. What invariant makes this unsafe operation valid?
2. Where is that invariant documented near the code or in a linked safety doc?
3. What local guard checks the invariant before the unsafe operation?
4. Which tests, Miri runs, sanitizer lanes, or hardware receipts exercise the
   seam?
5. If a witness is missing, is the follow-up tracked and is the current claim
   boundary honest?

If those answers are not in code, docs, policy, or receipts, the seam is not yet
review-fast.
