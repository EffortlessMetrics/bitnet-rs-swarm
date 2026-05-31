//! Local server generation stop control for timeout and cancellation receipts.

use serde::Serialize;
use std::time::Duration;

/// Stop reason recorded in dense local-server partial-generation receipts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalGenerationStopReason {
    Length,
    Stop,
    Timeout,
    Cancelled,
}

impl LocalGenerationStopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_partial(self) -> bool {
        matches!(self, Self::Timeout | Self::Cancelled)
    }
}

/// Request-level timeout and cancellation policy for one generation request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalGenerationControlPolicy {
    pub request_timeout: Duration,
    pub streaming: bool,
    pub cancel_after_tokens: Option<usize>,
}

impl LocalGenerationControlPolicy {
    pub fn dense_default() -> Self {
        Self {
            request_timeout: Duration::from_secs(300),
            streaming: false,
            cancel_after_tokens: None,
        }
    }
}

/// Stop decision emitted when generation must halt before normal completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct LocalGenerationStop {
    pub reason: LocalGenerationStopReason,
    pub stage: &'static str,
    pub generated_tokens: usize,
    pub elapsed_ms: u64,
}

/// Mutable controller used by token-boundary generation loops.
#[derive(Clone, Debug)]
pub struct LocalGenerationControl {
    policy: LocalGenerationControlPolicy,
    generated_tokens: usize,
    stop: Option<LocalGenerationStop>,
}

impl LocalGenerationControl {
    pub fn new(policy: LocalGenerationControlPolicy) -> Self {
        Self { policy, generated_tokens: 0, stop: None }
    }

    pub fn observe_prefill(&mut self, elapsed: Duration) -> Option<LocalGenerationStop> {
        self.check_timeout(elapsed, "prefill")
    }

    pub fn observe_decode_start(&mut self, elapsed: Duration) -> Option<LocalGenerationStop> {
        self.check_timeout(elapsed, "decode")
    }

    pub fn observe_token(&mut self, elapsed: Duration) -> Option<LocalGenerationStop> {
        self.generated_tokens = self.generated_tokens.saturating_add(1);
        if let Some(stop) = self.check_timeout(elapsed, "decode") {
            return Some(stop);
        }
        if self.policy.streaming
            && self.policy.cancel_after_tokens.is_some_and(|limit| self.generated_tokens >= limit)
        {
            let stop = LocalGenerationStop {
                reason: LocalGenerationStopReason::Cancelled,
                stage: "decode",
                generated_tokens: self.generated_tokens,
                elapsed_ms: elapsed.as_millis() as u64,
            };
            self.stop = Some(stop);
            return Some(stop);
        }
        None
    }

    pub fn complete(&mut self, reason: LocalGenerationStopReason, elapsed: Duration) {
        if self.stop.is_none() {
            self.stop = Some(LocalGenerationStop {
                reason,
                stage: "complete",
                generated_tokens: self.generated_tokens,
                elapsed_ms: elapsed.as_millis() as u64,
            });
        }
    }

    pub fn receipt(&self, request_id: &str, fallback_used: bool) -> LocalGenerationControlReceipt {
        let stop = self.stop.unwrap_or(LocalGenerationStop {
            reason: LocalGenerationStopReason::Length,
            stage: "complete",
            generated_tokens: self.generated_tokens,
            elapsed_ms: 0,
        });
        let timeout_reached = stop.reason == LocalGenerationStopReason::Timeout;
        let cancellation_observed = stop.reason == LocalGenerationStopReason::Cancelled;
        LocalGenerationControlReceipt {
            request_id: request_id.to_string(),
            requested_backend: "apple-m4-cpu-neon".to_string(),
            selected_backend: "apple-m4-cpu-neon".to_string(),
            fallback_used,
            generated_tokens: stop.generated_tokens,
            stop_reason: stop.reason.as_str().to_string(),
            partial_generation: stop.reason.is_partial(),
            timeout: LocalGenerationTimeoutReceipt {
                configured_ms: self.policy.request_timeout.as_millis() as u64,
                enforced: true,
                reached: timeout_reached,
                stage: timeout_reached.then_some(stop.stage.to_string()),
            },
            cancellation: LocalGenerationCancellationReceipt {
                cancellable: self.policy.streaming,
                requested: self.policy.cancel_after_tokens.is_some(),
                observed: cancellation_observed,
                stage: cancellation_observed.then_some(stop.stage.to_string()),
            },
            claim_boundary: LocalGenerationClaimBoundary {
                dense_slm_only: true,
                bitnet_serve_enabled: false,
                production_readiness_claimed: false,
                broad_quality_claimed: false,
                broad_speedup_claimed: false,
            },
        }
    }

    fn check_timeout(
        &mut self,
        elapsed: Duration,
        stage: &'static str,
    ) -> Option<LocalGenerationStop> {
        if elapsed < self.policy.request_timeout {
            return None;
        }
        let stop = LocalGenerationStop {
            reason: LocalGenerationStopReason::Timeout,
            stage,
            generated_tokens: self.generated_tokens,
            elapsed_ms: elapsed.as_millis() as u64,
        };
        self.stop = Some(stop);
        Some(stop)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalGenerationControlReceipt {
    pub request_id: String,
    pub requested_backend: String,
    pub selected_backend: String,
    pub fallback_used: bool,
    pub generated_tokens: usize,
    pub stop_reason: String,
    pub partial_generation: bool,
    pub timeout: LocalGenerationTimeoutReceipt,
    pub cancellation: LocalGenerationCancellationReceipt,
    pub claim_boundary: LocalGenerationClaimBoundary,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalGenerationTimeoutReceipt {
    pub configured_ms: u64,
    pub enforced: bool,
    pub reached: bool,
    pub stage: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalGenerationCancellationReceipt {
    pub cancellable: bool,
    pub requested: bool,
    pub observed: bool,
    pub stage: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalGenerationClaimBoundary {
    pub dense_slm_only: bool,
    pub bitnet_serve_enabled: bool,
    pub production_readiness_claimed: bool,
    pub broad_quality_claimed: bool,
    pub broad_speedup_claimed: bool,
}

pub fn health_ready_probe_receipt(
    health_elapsed: Duration,
    ready_elapsed: Duration,
    cheap_threshold: Duration,
) -> serde_json::Value {
    serde_json::json!({
        "health": {
            "generation_executed": false,
            "elapsed_ms": health_elapsed.as_millis() as u64,
            "cheap": health_elapsed <= cheap_threshold,
        },
        "ready": {
            "generation_executed": false,
            "elapsed_ms": ready_elapsed.as_millis() as u64,
            "cheap": ready_elapsed <= cheap_threshold,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m4_harden_timeout_enforces_partial_receipt_and_preserves_fallback_false() {
        let mut control = LocalGenerationControl::new(LocalGenerationControlPolicy {
            request_timeout: Duration::from_millis(5),
            streaming: false,
            cancel_after_tokens: None,
        });

        assert!(control.observe_prefill(Duration::from_millis(1)).is_none());
        let stop = control.observe_decode_start(Duration::from_millis(7)).expect("timeout stop");

        assert_eq!(stop.reason, LocalGenerationStopReason::Timeout);
        assert_eq!(stop.stage, "decode");

        let receipt = control.receipt("timeout-request", false);
        assert_eq!(receipt.stop_reason, "timeout");
        assert!(receipt.partial_generation);
        assert!(receipt.timeout.enforced);
        assert!(receipt.timeout.reached);
        assert!(!receipt.fallback_used);
        assert!(!receipt.claim_boundary.bitnet_serve_enabled);
    }

    #[test]
    fn m4_harden_cancel_records_streaming_stop_reason_and_partial_receipt() {
        let mut control = LocalGenerationControl::new(LocalGenerationControlPolicy {
            request_timeout: Duration::from_secs(30),
            streaming: true,
            cancel_after_tokens: Some(2),
        });

        assert!(control.observe_token(Duration::from_millis(1)).is_none());
        let stop = control.observe_token(Duration::from_millis(2)).expect("cancel stop");

        assert_eq!(stop.reason, LocalGenerationStopReason::Cancelled);
        assert_eq!(stop.generated_tokens, 2);

        let receipt = control.receipt("cancel-request", false);
        assert_eq!(receipt.stop_reason, "cancelled");
        assert_eq!(receipt.generated_tokens, 2);
        assert!(receipt.partial_generation);
        assert!(receipt.cancellation.cancellable);
        assert!(receipt.cancellation.requested);
        assert!(receipt.cancellation.observed);
        assert!(!receipt.fallback_used);
    }

    #[test]
    fn m4_harden_timeout_later_request_still_completes_and_health_ready_are_cheap() {
        let mut timed_out = LocalGenerationControl::new(LocalGenerationControlPolicy {
            request_timeout: Duration::from_millis(1),
            streaming: false,
            cancel_after_tokens: None,
        });
        assert!(timed_out.observe_decode_start(Duration::from_millis(2)).is_some());

        let mut later = LocalGenerationControl::new(LocalGenerationControlPolicy::dense_default());
        assert!(later.observe_token(Duration::from_millis(2)).is_none());
        later.complete(LocalGenerationStopReason::Length, Duration::from_millis(3));
        let receipt = later.receipt("later-request", false);

        assert_eq!(receipt.stop_reason, "length");
        assert_eq!(receipt.generated_tokens, 1);
        assert!(!receipt.partial_generation);
        assert!(!receipt.timeout.reached);
        assert!(!receipt.fallback_used);

        let probes = health_ready_probe_receipt(
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(10),
        );
        assert_eq!(probes["health"]["generation_executed"], false);
        assert_eq!(probes["ready"]["generation_executed"], false);
        assert_eq!(probes["health"]["cheap"], true);
        assert_eq!(probes["ready"]["cheap"], true);
    }
}
