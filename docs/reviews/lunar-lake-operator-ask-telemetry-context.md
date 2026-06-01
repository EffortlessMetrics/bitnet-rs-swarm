# Lunar Lake Operator Ask Telemetry Context Contract

Status: review
Owner: intel/openvino
Created: 2026-05-31
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](../specs/BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](../specs/BITNET-SPEC-OPENVINO-PHASE-TIMING.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: [#1110](https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1110)
Linked PRs: [#1116](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1116), [#1127](https://github.com/EffortlessMetrics/bitnet-rs-swarm/pull/1127)
Support-tier impact: no promotion; schema contract only
Policy impact: no policy exception

## Question

What telemetry context should successful `lunar-lake ask` receipts carry so an
operator can understand power and thermal context without treating that context
as `low_power` evidence?

## Decision

Successful `lunar-lake ask` receipts should carry a non-promotional
`telemetry_context` summary. The summary may be copied from a linked
`lunar-lake-power-thermal-context.json` receipt, sampled directly by the ask
command, or marked unavailable. It must never infer battery-mode evidence from
AC-only telemetry, missing thermal temperatures, or a route-profile aggregate.

The field is context for interpretation, not a route-promotion gate by itself.
It can make an answer receipt self-explanatory, but it cannot promote
`low_power`, claim power advantage, or claim measured temperatures.

Implementation status: #1116 landed this contract and #1127 added the
receipt-builder support for successful asks. Historical committed ask receipts
that predate #1127 may still omit `telemetry_context`; do not hand-refresh them
unless a real evidence rerun or a narrow validation issue requires it.

## Required Shape

Use this shape, or a versioned equivalent consumed by the receipt builder and
validator:

```json
{
  "telemetry_context": {
    "status": "linked",
    "source_receipt": "ci/hardware/intel-258v/2026-05-08/lunar-lake-power-thermal-context.json",
    "sample_scope": "current_machine_runtime_telemetry",
    "power": {
      "status": "sampled",
      "source": "os_power_probe",
      "power_scheme_raw": "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)",
      "power_scheme_guid": "381b4222-f694-41f0-9685-ff5bb260df2e",
      "power_scheme_name": "Balanced",
      "power_source": "ac",
      "battery_status_raw": "BatteryStatus=2;EstimatedChargeRemaining=100",
      "win32_battery_status_code": 2,
      "estimated_charge_remaining_percent": 100,
      "ac_power_inferred": true,
      "battery_mode_sample_recorded": false
    },
    "thermal": {
      "status": "zones_visible_values_unavailable",
      "source": "windows_perf_thermal_zone",
      "thermal_zones_visible": 1,
      "temperatures_celsius": [],
      "measured_temperature_available": false
    },
    "claim_boundary": {
      "low_power_evidence": false,
      "power_advantage_claim": false,
      "measured_temperature_claim": false,
      "route_promotion_changed": false
    }
  }
}
```

If no telemetry receipt is supplied and the ask command does not sample context,
the receipt should still carry:

```json
{
  "telemetry_context": {
    "status": "not_sampled",
    "source_receipt": null,
    "power": {
      "status": "not_sampled",
      "power_source": "unknown",
      "battery_mode_sample_recorded": false
    },
    "thermal": {
      "status": "not_sampled",
      "temperatures_celsius": [],
      "measured_temperature_available": false
    },
    "claim_boundary": {
      "low_power_evidence": false,
      "power_advantage_claim": false,
      "measured_temperature_claim": false,
      "route_promotion_changed": false
    }
  }
}
```

## Status Values

`telemetry_context.status`:

| Value | Meaning |
| --- | --- |
| `sampled` | The ask command sampled power/thermal context for this receipt. |
| `linked` | The ask receipt summarizes a linked telemetry receipt. |
| `not_sampled` | No telemetry source was queried for this ask receipt. |
| `not_exposed` | A source was queried but the required field was unavailable. |

`power.power_source`:

| Value | Meaning |
| --- | --- |
| `battery` | Battery/discharging state was observed and `ac_power_inferred=false`. |
| `ac` | AC, charging, or power-online state was observed or inferred. |
| `unknown` | The receipt cannot distinguish AC from battery honestly. |

`thermal.status`:

| Value | Meaning |
| --- | --- |
| `measured` | One or more usable temperature readings are present. |
| `zones_visible_values_unavailable` | Thermal zones are visible, but usable temperatures are absent. |
| `access_denied` | The thermal source exists but the OS denied readings. |
| `probe_unavailable` | No supported thermal source was available. |
| `not_sampled` | No thermal source was queried for this ask receipt. |
| `not_exposed` | The runtime or OS source does not expose the requested field. |

## Current Evidence Mapping

The committed telemetry receipt
`ci/hardware/intel-258v/2026-05-08/lunar-lake-power-thermal-context.json`
maps to:

| Field | Current value | Consequence |
| --- | --- | --- |
| `telemetry_context.status` | `linked` | Successful asks can cite this receipt as context. |
| `source_receipt` | `ci/hardware/intel-258v/2026-05-08/lunar-lake-power-thermal-context.json` | The ask receipt should keep the evidence path visible. |
| `power.power_source` | `ac` | This is not battery-mode evidence. |
| `power.power_scheme_guid` | `381b4222-f694-41f0-9685-ff5bb260df2e` | The raw Balanced scheme can be normalized. |
| `power.ac_power_inferred` | `true` | `low_power_evidence` must remain false. |
| `thermal.status` | `zones_visible_values_unavailable` | Thermal visibility exists, but measured-temperature claims remain blocked. |
| `thermal.temperatures_celsius` | `[]` | Do not invent temperatures or treat zero as measured data. |

The successful ask receipts that motivated #1110 omitted this summary. That
original omission is now a closed schema/builder gap for newly emitted asks. It
was not, and is still not, a reason to revoke current GPU profile routing,
because route-profile and regression aggregates already link the same telemetry
context and do not claim `low_power`.

## Fail-Closed Rules

Apply these rules before using the field in reviews, validators, or operator
summaries:

| Condition | Required handling |
| --- | --- |
| `status=not_sampled` | Keep the answer receipt valid, but report power/thermal context as unavailable. |
| `power_source=ac` or `ac_power_inferred=true` | Set `low_power_evidence=false`; do not use the receipt for POWER-006 battery samples. |
| `power_source=unknown` | Treat low-power and power-advantage evidence as blocked. |
| `battery_mode_sample_recorded=false` | Do not count the ask as a battery route sample. |
| `thermal.status` is not `measured` | Preserve route evidence, but block measured-temperature claims. |
| `temperatures_celsius=[]` | Do not coerce missing values to zero or ambient temperature. |
| `source_receipt` is missing while `status=linked` | Treat the telemetry summary as invalid. |
| `power_advantage_claim=true` in an ask receipt | Reject unless a separate promotion review links benchmark-qualified power evidence. |

## Implementation Acceptance For #1110

The schema/builder PR for #1110 closed in #1127 after it:

- added `telemetry_context` to successful `lunar-lake ask` receipts;
- included `source_receipt` when the context comes from
  `lunar-lake-power-thermal-context.json` or an equivalent telemetry receipt;
- normalized `power_scheme_guid` and `power_scheme_name` when a Windows
  `powercfg /getactivescheme` string is available;
- preserved AC, battery, and unknown power states explicitly;
- preserved thermal unavailability without invented temperatures;
- added focused receipt-builder tests for linked, `not_sampled`, and
  unavailable thermal contexts;
- kept blocked low-power asks and POWER-006 route samples governed by the
  stricter battery evidence contract in
  [lunar-lake-power-telemetry.md](../research/lunar-lake-power-telemetry.md).

Future telemetry-context work should not rerun inference, refresh benchmark
matrices, change route policy, or edit generated dashboards unless a new,
narrow evidence issue defines that scope.

## Claim Boundary

This contract does not add:

- new Lunar Lake inference;
- route-policy mutation;
- route promotion or revocation;
- battery-mode `low_power` evidence;
- power-advantage evidence;
- measured-temperature evidence;
- speedup or acceleration claims;
- native OpenCL or native NPU proof;
- BitNet QK256/I2_S behavior proof.

It only defines the successful-ask telemetry summary that #1110/#1127 added
without spending runtime or CI budget on a broader implementation.
