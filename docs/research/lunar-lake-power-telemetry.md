# Lunar Lake Power Telemetry Research

Research issue: https://github.com/EffortlessMetrics/bitnet-rs-swarm/issues/1033

Research date: 2026-05-30

Repository: `EffortlessMetrics/bitnet-rs-swarm`

## Executive Summary

`LNL258V-POWER-006` should not move directly into another broad evidence PR.
The first useful step is to tighten the telemetry model for battery-mode route
samples.

Current evidence says:

- `Win32_Battery` is enough for a strict AC-vs-battery preflight. The existing
  CLI correctly fails closed when `BatteryStatus=2/6/7/8/9/11` or
  `ac_power_inferred=true`.
- `root\wmi` battery classes expose richer capacity fields on this laptop than
  the CLI currently records. They should be the next source for the energy
  proxy, with `Win32_Battery.EstimatedChargeRemaining` kept as the coarse
  fallback.
- Thermal classes are visible but weak on this host. Current results show a
  thermal zone count, but measured temperatures are unavailable or zero. That
  is usable as thermal availability context, not measured-temperature evidence.
- `powercfg /batteryreport` is useful for manual sanity checks and installed
  battery metadata, but the raw HTML includes device-identifying fields and is
  not the right committed route receipt format.
- OpenVINO can identify CPU/GPU/NPU devices and cache/performance knobs, but the
  local property set does not provide direct route-level watts, battery drain,
  or temperature readings. OpenVINO data should support device identity and
  cache-mode evidence, not power claims.

Recommended next implementation is small: extend the low-power telemetry
receipt schema to record normalized battery capacity fields from `root\wmi`,
the existing power scheme and `Win32_Battery` fields, and explicit thermal
availability status. Do not run route samples until that schema is clear and
the `POWER-006` allowed paths include the sample receipts named by the runbook.

## Current Repo Behavior

The existing implementation records:

- memory via `sysinfo`;
- active Windows power scheme via `powercfg /GETACTIVESCHEME`;
- battery state via `Get-CimInstance Win32_Battery`;
- thermal temperatures via `MSAcpi_ThermalZoneTemperature` when accessible;
- thermal-zone visibility via
  `Win32_PerfFormattedData_Counters_ThermalZoneInformation` when temperature
  values are unavailable.

Relevant source:

- `crates/bitnet-cli/src/commands/lunar_lake.rs`:
  - `TelemetryPowerContext`
  - `TelemetryThermalContext`
  - `collect_telemetry_power_context`
  - `collect_telemetry_thermal_context`
  - `infer_ac_power_from_battery_status`

The committed run-harness receipt currently records an AC-blocked preflight:

- `ci/hardware/intel-258v/2026-05-08/lunar-lake-low-power-run-harness.json`
- `battery_preflight_passed=false`
- `model_inference_allowed=false`
- `model_inference_executed=false`
- `route_sample_execution_started=false`
- `battery_status=BatteryStatus=2;EstimatedChargeRemaining=100`
- `ac_power_inferred=true`

During this research session, the strict preflight was observed to pass while
Windows reported battery mode, then later fail closed after the system returned
to AC:

| Time UTC | Command | Result | Key fields |
| --- | --- | --- | --- |
| 2026-05-30T18:38:36Z | `telemetry-context --require-battery --strict` | passed | `BatteryStatus=1`, charge 98%, `ac_power_inferred=false`, `requirement_satisfied=true` |
| 2026-05-30T18:43:35Z | `telemetry-context --require-battery --strict` | failed closed | `BatteryStatus=2`, charge 97%, `ac_power_inferred=true`, `requirement_satisfied=false` |
| 2026-05-30T18:43:35Z | `low-power-harness --strict` | failed closed | `battery_preflight_passed=false`, `model_inference_allowed=false` |

That behavior is correct. It also means the battery run needs a stable
preflight/start/end receipt sequence, not one opportunistic point sample.

## Source Findings

### `Win32_Battery`

Command:

```powershell
Get-CimInstance -ClassName Win32_Battery |
  Select-Object Name,BatteryStatus,EstimatedChargeRemaining,Status,
    PowerManagementSupported,DesignCapacity,FullChargeCapacity,
    TimeOnBattery,EstimatedRunTime
```

Observed on this host:

- `BatteryStatus=1` when discharging was briefly observed.
- `BatteryStatus=2` when AC was later observed.
- `EstimatedChargeRemaining` was available.
- `Status=OK` was available.
- `DesignCapacity` and `FullChargeCapacity` were empty through
  `Win32_Battery` on this host.
- `EstimatedRunTime` returned `71582788` while AC was present; do not use that
  value as route evidence without a sanity guard.

Use:

- Keep as the primary strict preflight source for AC-vs-battery.
- Keep `EstimatedChargeRemaining` as a coarse percent field.
- Do not rely on `DesignCapacity`, `FullChargeCapacity`, or
  `EstimatedRunTime` from this class on this host.

Microsoft documents `Win32_Battery.BatteryStatus`, including `1` for
discharging and `2` for AC access with no battery discharge, and documents
`EstimatedChargeRemaining` as percent.

### `root\wmi` Battery Classes

Commands:

```powershell
Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus |
  Select-Object InstanceName,PowerOnline,Discharging,Charging,
    RemainingCapacity,Voltage,Rate,EstimatedRuntime

Get-CimInstance -Namespace root\wmi -ClassName BatteryFullChargedCapacity |
  Select-Object InstanceName,FullChargedCapacity
```

Observed on this host while AC was present:

- `PowerOnline=True`
- `Discharging=False`
- `Charging=False`
- `RemainingCapacity=68350`
- `Voltage=8580`
- `Rate` empty
- `EstimatedRuntime` empty
- `FullChargedCapacity=70000`

`BatteryStaticData` returned a generic failure on this host.

Use:

- Add these fields to the before/after telemetry schema when available.
- Prefer `RemainingCapacity` and `FullChargedCapacity` over percent-only
  charge deltas for energy-proxy normalization.
- Record `PowerOnline`, `Discharging`, and `Charging` alongside
  `Win32_Battery.BatteryStatus` to catch ambiguous status values.
- Treat `Rate`, `EstimatedRuntime`, and static design fields as optional.

### `powercfg /GETACTIVESCHEME`

Command:

```powershell
powercfg /getactivescheme
```

Observed:

```text
Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)
```

Use:

- Keep recording the active scheme string in every preflight, before, after,
  and route sample receipt.
- Add a normalized `power_scheme_guid` and `power_scheme_name` if the schema is
  revised, so comparisons do not depend on parsing a display string later.

### `powercfg /batteryreport`

Command:

```powershell
powercfg /batteryreport /output target\research\lunar-lake-batteryreport.html /duration 1
```

Observed:

- Report generation succeeded.
- Installed battery metadata included design capacity, full charge capacity,
  chemistry, cycle count, and recent AC/battery usage sections.

Use:

- Use for manual sanity checks and capacity cross-checks.
- Do not commit raw battery-report HTML as route evidence. It is not the
  receipt schema, may include device-identifying fields, and is awkward to diff.
- If needed later, parse a redacted subset into a structured receipt field:
  `battery_report_summary`.

### Windows Power API Candidate

Candidate source:

- `GetSystemPowerStatus`
- `SYSTEM_POWER_STATUS`

Use:

- Consider adding a small Windows API probe if WMI values keep disagreeing.
- Candidate fields: AC line status, battery flag, battery life percent, battery
  life time, full life time, and battery saver status.
- This should supplement, not replace, WMI battery receipts until it is tested
  on the 258V host.

### Thermal Classes

Commands:

```powershell
Get-CimInstance -Namespace root\wmi -ClassName MSAcpi_ThermalZoneTemperature |
  Select-Object InstanceName,CurrentTemperature,CriticalTripPoint,PassiveTripPoint

Get-CimInstance -ClassName Win32_TemperatureProbe |
  Select-Object Name,Status,CurrentReading

Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation |
  Select-Object Name,HighPrecisionTemperature,Temperature,PercentPassiveLimit
```

Observed:

- `MSAcpi_ThermalZoneTemperature` returned access denied.
- `Win32_TemperatureProbe` returned one `Numeric Sensor` with
  `Status=Unknown` and no current reading.
- `Win32_PerfFormattedData_Counters_ThermalZoneInformation` returned
  `\_TZ.TZ00`, but `HighPrecisionTemperature=0` and `Temperature=0`.
- The CLI correctly records `windows_perf_thermal_zone` with a visible zone and
  an empty `temperatures_celsius` list.

Use:

- Keep `thermal_zones_visible` and `thermal_sensor_status`.
- Treat zero or empty temperature values as unavailable.
- Do not block battery route sampling only because measured thermal readings
  are unavailable, but block any measured-temperature claim.

### OpenVINO Device Properties

Command:

```powershell
$script = @'
import json
import openvino as ov
core = ov.Core()
print(json.dumps({
  "version": ov.__version__,
  "available_devices": list(core.available_devices),
  "devices": {
    str(dev): {
      "FULL_DEVICE_NAME": str(core.get_property(dev, "FULL_DEVICE_NAME")),
      "SUPPORTED_PROPERTIES": str(core.get_property(dev, "SUPPORTED_PROPERTIES")),
      "OPTIMIZATION_CAPABILITIES": [str(v) for v in core.get_property(dev, "OPTIMIZATION_CAPABILITIES")],
      "DEVICE_ARCHITECTURE": str(core.get_property(dev, "DEVICE_ARCHITECTURE")),
    }
    for dev in core.available_devices
  },
}, indent=2))
'@
$script | python -
```

Observed:

- OpenVINO version: `2026.1.0-21367-63e31528c62-releases/2026/1`
- Available devices: `CPU`, `GPU`, `NPU`
- CPU full name: `Intel(R) Core(TM) Ultra 7 258V`
- GPU full name: `Intel(R) Arc(TM) 140V GPU (16GB) (iGPU)`
- NPU full name: `Intel(R) AI Boost`
- GPU properties include cache and performance knobs such as `CACHE_DIR`,
  `CACHE_MODE`, `PERFORMANCE_HINT`, `GPU_QUEUE_THROTTLE`, and memory statistics.
- NPU properties include cache/performance/turbo-related knobs such as
  `CACHE_DIR`, `CACHE_MODE`, `PERFORMANCE_HINT`, `NPU_TURBO`,
  `NPU_DRIVER_VERSION`, `NPU_DEVICE_TOTAL_MEM_SIZE`, and
  `NPU_DEVICE_ALLOC_MEM_SIZE`.
- The queried properties did not expose direct watts, battery drain, or
  temperature values.

Use:

- Record OpenVINO device identity in route receipts.
- Record cache/performance/turbo configuration when relevant to route timing.
- Do not use OpenVINO properties as direct low-power evidence unless a later
  probe proves a measured power field exists and is stable.

## Recommended Receipt Schema

### Preflight Receipt

Required fields:

- `created_utc`
- `machine_id`
- `power_scheme_raw`
- `power_scheme_guid`
- `power_scheme_name`
- `win32_battery_status_code`
- `win32_battery_status_label`
- `estimated_charge_percent`
- `ac_power_inferred`
- `battery_mode_required`
- `battery_mode_sample_recorded`
- `requirement_satisfied`
- `model_inference_allowed`
- `model_inference_executed=false`
- `route_sample_execution_started=false`
- `thermal_sensor_status`
- `thermal_zones_visible`
- `temperatures_celsius`
- `gaps`
- `claim_boundary`

Optional fields:

- `system_power_status`
- `wmi_battery_status`
- `battery_report_summary`
- `openvino_device_summary`

### Before/After Battery Telemetry Receipts

Required fields:

- all preflight power and thermal fields;
- `sample_role`: `before` or `after`;
- `battery_status_source_set`: list of source names captured;
- `wmi_power_online`
- `wmi_discharging`
- `wmi_charging`
- `wmi_remaining_capacity_mwh_or_raw`
- `wmi_full_charged_capacity_mwh_or_raw`
- `wmi_voltage_mv`
- `wmi_rate_raw`
- `elapsed_since_preflight_ms` if available.

The `mwh_or_raw` suffix is deliberate until the units are verified from a
stable Microsoft or ACPI battery-class reference for this exact class. The
energy proxy can still compare like-for-like deltas if the source and units are
recorded explicitly.

### Route Sample Receipts

Required additions for `POWER-006` route samples:

- `route_id`
- `profile`
- `power_mode`
- `power_scheme_guid`
- `battery_status`
- `estimated_charge_before`
- `estimated_charge_after`
- `wmi_remaining_capacity_before`
- `wmi_remaining_capacity_after`
- `elapsed_wall_ms`
- `first_token_ms`
- `decode_total_ms`
- `tokens_per_second`
- `total_response_ms`
- `answer_gate_passed`
- `fallback_used=false`
- `requested_backend`
- `selected_backend`
- `runtime_api`
- `selected_kernel_or_runtime`
- `openvino_device_summary` for OpenVINO routes
- `thermal_context`
- `power_context`
- `route_decision_impact`

### Energy Proxy Receipt

Required fields:

- `before_telemetry_context`
- `after_telemetry_context`
- `sample_count`
- `route`
- `profile`
- `elapsed_wall_ms`
- `charge_delta_percent`
- `remaining_capacity_delta_raw`
- `capacity_source`
- `proxy_kind`: `remaining_capacity_delta`, `percent_delta`, or `blocked`
- `valid_battery_mode_evidence`
- `power_advantage_claim=false` unless refreshed benchmark evidence qualifies
  it.

## Promotion Blockers

`low_power` must remain unpromoted while any of these are true:

- preflight or before/after receipts report AC, charging, unknown, or
  `ac_power_inferred=true`;
- before and after samples are not both battery-mode samples;
- route sample receipts are missing for CPU, OpenVINO GPU, or OpenVINO NPU;
- any route sample uses hidden fallback or omits selected backend/runtime
  identity;
- answer gates fail for the route/profile under consideration;
- timing is unstable or not profile-matched;
- energy proxy is percent-only with too little elapsed time or sample count to
  separate route behavior from noise;
- no route has benchmark-qualified power advantage over the current default;
- strict regression or operator comparison cannot preserve the same decision.

Thermal temperature unavailability should remain a gap, not by itself a blocker
for collecting route samples. It blocks measured-temperature claims.

## Tracker Scope Finding

`LNL258V-POWER-006` currently names route sample receipts in the runbook, but
the work-item `allowed_paths` list does not include those files:

- `lunar-lake-operator-ask-battery-low-power-cpu.json`
- `lunar-lake-operator-ask-battery-low-power-gpu.json`
- `lunar-lake-operator-ask-battery-low-power-npu.json`

Before executing the physical route matrix, create a small tracker-scope repair
or issue update that adds the intended sample receipt paths or replaces the
three loose receipts with one explicitly allowed battery route-sample bundle.

## Recommended Next Steps

1. Add a narrow schema-hardening issue or PR for the richer Windows battery
   fields from `root\wmi`.
2. Add `power_scheme_guid` and `power_scheme_name` normalization.
3. Add thermal status normalization that distinguishes:
   `measured_temperatures`, `zones_visible_values_unavailable`,
   `access_denied`, and `probe_unavailable`.
4. Repair the `POWER-006` allowed paths before running route samples.
5. Only then run the battery preflight, before receipt, CPU/GPU/NPU route
   samples, after receipt, energy proxy, and artifact refresh sequence.

## References

- Microsoft Learn: `Win32_Battery` class,
  https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-battery
- Microsoft Learn: `powercfg` command-line options,
  https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/powercfg-command-line-options
- Microsoft Learn: `GetSystemPowerStatus`,
  https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getsystempowerstatus
- Microsoft Learn: `SYSTEM_POWER_STATUS`,
  https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-system_power_status
- OpenVINO documentation: query device properties,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html
- OpenVINO documentation: NPU device,
  https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
