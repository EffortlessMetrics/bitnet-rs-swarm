#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    dim: u8,
    seq_len: u8,
    base: f32,
    /// Optional base for `resolve_base` – exercises both Some and None paths.
    resolve_base_arg: Option<f32>,
}

fuzz_target!(|input: Input| {
    use bitnet_rope::{build_tables, resolve_base};

    // Exercise resolve_base with both None and Some variants.
    let _resolved_none = resolve_base(None);
    let _resolved_some = resolve_base(input.resolve_base_arg);

    let dim = input.dim as usize;
    let seq_len = input.seq_len as usize;

    if let Ok(tables) = build_tables(dim, seq_len, input.base) {
        // Shape invariant: both vecs must have exactly seq_len * half_dim entries.
        assert_eq!(tables.sin.len(), seq_len * tables.half_dim);
        assert_eq!(tables.cos.len(), seq_len * tables.half_dim);

        // Trig identity: sin² + cos² ≈ 1 for every pair — but only when every
        // angle pos * inv_freq is a finite f32. inv_freq = base^(-2i/dim)
        // grows without bound as base -> 0+, so a tiny positive base (CI
        // crash-736af2db: dim=254, seq_len=255, base=8.22931e-40) overflows
        // inv_freq (or the angle) to +inf, and sin/cos of a non-finite value
        // is NaN by IEEE-754. That is an f32 domain limit, not a
        // table-generation bug, so gate the identity on the largest possible
        // angle: (seq_len-1) * max(1, base^(-(dim-2)/dim)).
        let max_inv_freq = 1.0 / input.base.powf((dim as f32 - 2.0) / dim as f32);
        let max_angle = seq_len.saturating_sub(1) as f32 * max_inv_freq.max(1.0);
        if max_angle.is_finite() {
            for (s, c) in tables.sin.iter().zip(&tables.cos) {
                let norm = s * s + c * c;
                assert!((norm - 1.0).abs() < 1e-5, "sin²+cos²={norm} != 1.0 (sin={s}, cos={c})");
            }
        }
    }
});
