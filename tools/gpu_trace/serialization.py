from __future__ import annotations

import json
from enum import Enum
from typing import Any

import cattrs
from cattrs.strategies import configure_tagged_union

from .models import (
    CounterType,
    GpuCounterDefinition,
    GpuPerformanceStateInterval,
    GpuUtilization,
    KernelDispatch,
    KernelStats,
    PerformanceState,
    ShaderInfo,
    ShaderType,
    TraceExport,
    TraceMetadata,
)


def _make_converter() -> cattrs.Converter:
    converter = cattrs.Converter()

    def enum_unstructure(val: Enum) -> str:
        return val.value

    def enum_structure(val: str, cls: type[Enum]) -> Enum:
        return cls(val)

    for enum_cls in (CounterType, PerformanceState, ShaderType):
        converter.register_unstructure_hook(enum_cls, enum_unstructure)
        converter.register_structure_hook(enum_cls, enum_structure)

    def tuple_unstructure(val: tuple[Any, ...]) -> list[Any]:
        return [converter.unstructure(item) for item in val]

    converter.register_unstructure_hook(tuple, tuple_unstructure)

    return converter


_converter = _make_converter()


def to_dict(export: TraceExport) -> dict[str, Any]:
    data = _converter.unstructure(export)
    data.pop("counter_samples", None)
    return data


def to_json(export: TraceExport, indent: int | None = 2) -> str:
    return json.dumps(to_dict(export), indent=indent)


def from_dict(data: dict[str, Any]) -> TraceExport:
    # Validate the input before cattrs can coerce strings, booleans or floats.
    utilization = data.get("utilization")
    if not isinstance(utilization, dict) or any(
        type(utilization.get(name)) is not int or not 0 <= utilization[name] <= 2**63 - 1
        for name in GpuUtilization.__dataclass_fields__
    ):
        raise ValueError("Invalid GPU utilization durations or counts")
    duration = utilization["run_duration_ns"]
    if (
        utilization["global_active_ns"] + utilization["global_idle_ns"] > utilization["global_coverage_ns"]
        or utilization["global_coverage_ns"] > duration
        or utilization["target_active_ns"] > utilization["global_active_ns"]
        or utilization["window_start_ns"] > utilization["window_end_ns"]
        or utilization["window_end_ns"] > duration
        or utilization["target_active_ns"] > utilization["window_end_ns"] - utilization["window_start_ns"]
    ):
        raise ValueError("Inconsistent GPU utilization durations")
    export = _converter.structure(data, TraceExport)
    if type(data.get("metadata", {}).get("duration_ns")) is not int or export.metadata.duration_ns < 0:
        raise ValueError("Invalid trace duration")
    return export


def from_json(json_str: str) -> TraceExport:
    return from_dict(json.loads(json_str))
