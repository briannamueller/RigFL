"""Resource accounting for experiment runners."""

from __future__ import annotations

import contextlib
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn

COMMUNICATION_KINDS = (
    "client_to_server",
    "server_to_client",
    "peer_to_peer",
)
CACHED_MEASUREMENT_KIND = "rigfl.cached_resource"
CACHED_MEASUREMENT_SCHEMA_VERSION = 1


def payload_bytes(value: Any, _seen: set[int] | None = None) -> int:
    """Return the logical payload size, excluding protocol metadata."""
    seen = set() if _seen is None else _seen
    if value is None:
        return 0
    marker = id(value)
    if marker in seen:
        return 0
    seen.add(marker)

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, memoryview):
        return value.nbytes
    if isinstance(value, nn.Module):
        return payload_bytes(value.state_dict(), seen)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(payload_bytes(getattr(value, field.name), seen)
                   for field in fields(value))
    if isinstance(value, Mapping):
        return sum(payload_bytes(item, seen) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)):
        return sum(payload_bytes(item, seen) for item in value)
    return 0


def write_cached_measurement(path: Path, *, fingerprint: str,
                             measurement: dict) -> None:
    from rigfl.experiment.artifacts import atomic_write_json
    atomic_write_json(Path(path), {
        "kind": CACHED_MEASUREMENT_KIND,
        "schema_version": CACHED_MEASUREMENT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        **measurement,
    })


def load_cached_measurement(path: Path, *, fingerprint: str) -> dict | None:
    try:
        saved = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict):
        return None
    if saved.get("kind") != CACHED_MEASUREMENT_KIND:
        return None
    if saved.get("schema_version") != CACHED_MEASUREMENT_SCHEMA_VERSION:
        return None
    if saved.get("fingerprint") != fingerprint:
        return None
    wall_seconds = saved.get("wall_seconds")
    flops = saved.get("flops")
    measurement = saved.get("measurement")
    valid_wall = (isinstance(wall_seconds, (int, float))
                  and not isinstance(wall_seconds, bool)
                  and wall_seconds >= 0 and math.isfinite(wall_seconds))
    valid_flops = (flops is None or
                   (isinstance(flops, int) and not isinstance(flops, bool)
                    and flops >= 0))
    if not valid_wall or not valid_flops or not isinstance(measurement, dict):
        return None
    if not isinstance(measurement.get("hardware"), dict):
        return None
    if not isinstance(measurement.get("flop_estimation"), dict):
        return None
    return saved


def _hardware_signature(device: torch.device) -> dict:
    cpu_name, cpu_source = _cpu_identity()
    signature = {
        "device_type": device.type,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_name": cpu_name,
        "cpu_identity_source": cpu_source,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        signature["device_name"] = torch.cuda.get_device_name(index)
        signature["device_memory_bytes"] = properties.total_memory
        signature["compute_capability"] = (
            f"{properties.major}.{properties.minor}")
        signature["multiprocessor_count"] = properties.multi_processor_count
    else:
        signature["device_name"] = cpu_name
    return signature


@lru_cache(maxsize=1)
def _cpu_identity() -> tuple[str, str]:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip(), "proc_cpuinfo"
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            found = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False, capture_output=True, text=True,
            ).stdout.strip()
            if found:
                return found, "sysctl"
        except OSError:
            pass
        try:
            details = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                check=False, capture_output=True, text=True,
            ).stdout.splitlines()
            for line in details:
                label, separator, value = line.strip().partition(":")
                if separator and label == "Chip" and value.strip():
                    return value.strip(), "system_profiler"
        except OSError:
            pass
    fallback = platform.processor() or platform.machine()
    return fallback, "generic_fallback"


def _hardware_identity_complete(signature: dict) -> bool:
    return signature.get("cpu_identity_source") != "generic_fallback"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@dataclass
class ResourceDelta:
    """Measurements captured for one completed block."""

    wall_seconds: float = 0.0
    flops: int | None = None


class ResourceMonitor:
    """Collect operation timing, estimated FLOPs, and logical transfers."""

    def __init__(self, device: torch.device, *, estimate_flops: bool = False,
                 clock=time.perf_counter):
        self.device = device
        self.estimate_flops = estimate_flops
        self.clock = clock
        self.hardware = _hardware_signature(device)
        self._counter = None
        self._run_started = None
        self._run_wall_seconds = 0.0
        self._operations: dict[str, dict] = {}
        self._clients: dict[str, dict] = {}
        self._communication = {name: 0 for name in COMMUNICATION_KINDS}
        self._checkpoints: list[dict] = []
        self._reused: list[dict] = []
        self._missing_reused: list[str] = []
        self._cache_status: dict[str, list[str]] = defaultdict(list)

    def __enter__(self):
        if self.estimate_flops:
            try:
                from torch.utils.flop_counter import FlopCounterMode
            except ImportError as exc:
                raise RuntimeError(
                    "estimate_flops requires PyTorch 2.1 or newer, with "
                    "torch.utils.flop_counter.FlopCounterMode"
                ) from exc
            self._counter = FlopCounterMode(display=False)
            self._counter.__enter__()
        _sync(self.device)
        self._run_started = self.clock()
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync(self.device)
        self._run_wall_seconds = self.clock() - self._run_started
        if self._counter is not None:
            self._counter.__exit__(exc_type, exc, tb)

    def _flops(self) -> int | None:
        return (None if self._counter is None
                else int(self._counter.get_total_flops()))

    @contextlib.contextmanager
    def capture(self):
        delta = ResourceDelta()
        _sync(self.device)
        started = self.clock()
        initial_flops = self._flops()
        yield delta
        _sync(self.device)
        delta.wall_seconds = self.clock() - started
        final_flops = self._flops()
        if initial_flops is not None and final_flops is not None:
            delta.flops = final_flops - initial_flops

    @contextlib.contextmanager
    def measure(self, operation: str, *, category: str,
                client_id: int | None = None):
        with self.capture() as delta:
            yield
        slot = self._operations.setdefault(operation, {
            "category": category,
            "calls": 0,
            "wall_seconds": 0.0,
            "flops": 0 if self.estimate_flops else None,
        })
        slot["calls"] += 1
        slot["wall_seconds"] += delta.wall_seconds
        if delta.flops is not None:
            slot["flops"] += delta.flops
        if client_id is not None:
            client = self._client(client_id)
            client_category = (
                "algorithm_operations" if category == "algorithm" else "evaluation"
            )
            client["wall_seconds"][client_category] += delta.wall_seconds
            if delta.flops is not None:
                client["flops"][client_category] += delta.flops

    def record_transfer(self, kind: str, size: int, *,
                        sender: int | None = None,
                        receiver: int | None = None) -> None:
        if kind not in self._communication:
            raise ValueError(f"unknown communication kind: {kind}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("communication size must be a non-negative integer")
        self._communication[kind] += size
        if sender is not None:
            client = self._client(sender)
            client["sent_bytes"] += size
        if receiver is not None:
            client = self._client(receiver)
            client["received_bytes"] += size

    def _client(self, client_id: int) -> dict:
        return self._clients.setdefault(str(client_id), {
            "wall_seconds": {"algorithm_operations": 0.0, "evaluation": 0.0},
            "flops": {
                "algorithm_operations": 0 if self.estimate_flops else None,
                "evaluation": 0 if self.estimate_flops else None,
            },
            "sent_bytes": 0,
            "received_bytes": 0,
        })

    def checkpoint(self, round_idx: int) -> None:
        operation_seconds = sum(
            values["wall_seconds"] for values in self._operations.values()
            if values["category"] == "algorithm"
        )
        operation_flops = None
        if self.estimate_flops:
            operation_flops = sum(
                values["flops"] for values in self._operations.values()
                if values["category"] == "algorithm"
            )
        self._checkpoints.append({
            "round": round_idx,
            "algorithm_wall_seconds": operation_seconds,
            "algorithm_flops": operation_flops,
            "communication_bytes": sum(self._communication.values()),
        })

    def add_reused(self, name: str, measurement: dict | None, *,
                   access: ResourceDelta | None = None) -> None:
        if measurement is None:
            self._missing_reused.append(name)
            return
        current_access = {
            "wall_seconds": 0.0 if access is None else access.wall_seconds,
            "flops": (0 if access is None and self.estimate_flops
                      else None if access is None else access.flops),
        }
        self._reused.append({
            "name": name,
            **measurement,
            "current_access": current_access,
        })

    def record_cache(self, stage: str, status: str) -> None:
        if status not in {"hit", "miss", "disabled"}:
            raise ValueError(f"unknown cache status: {status}")
        self._cache_status[stage].append(status)

    def measurement_signature(self) -> dict:
        return {
            "hardware": self.hardware,
            "flop_estimation": {
                "enabled": self.estimate_flops,
                "method": ("torch.utils.flop_counter.FlopCounterMode"
                           if self.estimate_flops else None),
                "convention": ("multiply-add=2" if self.estimate_flops else None),
                "torch_version": str(torch.__version__),
            },
        }

    def artifact_measurement(self, delta: ResourceDelta) -> dict:
        return {
            "wall_seconds": delta.wall_seconds,
            "flops": delta.flops,
            "measurement": self.measurement_signature(),
        }

    def to_dict(self) -> dict:
        algorithm = [v for v in self._operations.values()
                     if v["category"] == "algorithm"]
        evaluation = [v for v in self._operations.values()
                      if v["category"] == "evaluation"]
        observed_algorithm_flops = (
            sum(v["flops"] for v in algorithm)
            if self.estimate_flops else None
        )
        observed_evaluation_flops = (
            sum(v["flops"] for v in evaluation)
            if self.estimate_flops else None
        )
        flop_signature = self.measurement_signature()["flop_estimation"]
        reused_flops = None
        access_flops = None
        compatible_reused_flops = all(
            item.get("measurement", {}).get("flop_estimation") == flop_signature
            for item in self._reused
        )
        if (self.estimate_flops and not self._missing_reused
                and compatible_reused_flops):
            values = [item.get("flops") for item in self._reused]
            access_values = [item["current_access"].get("flops")
                             for item in self._reused]
            if (all(value is not None for value in values)
                    and all(value is not None for value in access_values)):
                reused_flops = sum(values)
                access_flops = sum(access_values)
        observed_algorithm_seconds = sum(v["wall_seconds"] for v in algorithm)
        observed_evaluation_seconds = sum(v["wall_seconds"] for v in evaluation)
        reused_seconds = sum(item.get("wall_seconds", 0.0) for item in self._reused)
        access_seconds = sum(
            item["current_access"]["wall_seconds"] for item in self._reused)
        same_hardware = (
            not self._missing_reused
            and (not self._reused or (
                _hardware_identity_complete(self.hardware)
                and all(item.get("measurement", {}).get("hardware") == self.hardware
                        for item in self._reused)
            ))
        )
        attributed_seconds = (
            max(0.0, observed_algorithm_seconds - access_seconds) + reused_seconds
            if same_hardware else None
        )
        attributed_flops = (
            max(0, observed_algorithm_flops - access_flops) + reused_flops
            if (observed_algorithm_flops is not None
                and reused_flops is not None and access_flops is not None)
            else None
        )
        communication = dict(self._communication)
        communication["total"] = sum(self._communication.values())
        return {
            "schema_version": 1,
            "measurement": {
                "communication": {
                    "basis": "logical algorithm payload bytes",
                    "includes_protocol_metadata": False,
                },
                "timing": {
                    "clock": "time.perf_counter",
                    "accelerator_synchronized": self.device.type in {"cuda", "mps"},
                    "hardware": self.hardware,
                },
                "flop_estimation": flop_signature,
            },
            "observed": {
                "communication_bytes": communication,
                "flops": {
                    "algorithm_operations": observed_algorithm_flops,
                    "evaluation": observed_evaluation_flops,
                    "total": (None if observed_algorithm_flops is None else
                              observed_algorithm_flops + observed_evaluation_flops),
                },
                "wall_seconds": {
                    "algorithm_operations": observed_algorithm_seconds,
                    "evaluation": observed_evaluation_seconds,
                    "total": self._run_wall_seconds,
                },
            },
            "reused": self._reused,
            "missing_reused_measurements": self._missing_reused,
            "cache_reuse": {
                stage: _cache_summary(statuses)
                for stage, statuses in self._cache_status.items()
            },
            "attributed_training": {
                "flops": attributed_flops,
                "wall_seconds": attributed_seconds,
                "wall_seconds_comparable": same_hardware,
            },
            "operations": self._operations,
            "clients": self._clients,
            "checkpoints": self._checkpoints,
        }


def _cache_summary(statuses: list[str]) -> str:
    if statuses and all(status == "disabled" for status in statuses):
        return "disabled"
    hits = sum(status == "hit" for status in statuses)
    if hits == 0:
        return "none"
    return "complete" if hits == len(statuses) else "partial"


@contextlib.contextmanager
def measured(monitor: ResourceMonitor | None, operation: str, *, category: str,
             client_id: int | None = None):
    if monitor is None:
        yield
        return
    with monitor.measure(operation, category=category, client_id=client_id):
        yield
