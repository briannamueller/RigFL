"""Utilities for nested experiment configuration paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel


def nested_get(mapping: dict, path: str, default=None):
    value = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def nested_set(mapping: dict, path: str, value) -> None:
    parts = path.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"{'.'.join(parts[:-1])} is not a configuration mapping")
        current = child
    current[parts[-1]] = value


def nested_delete(mapping: dict, path: str) -> None:
    parts = path.split(".")
    current = mapping
    parents = []
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        parents.append((current, part))
        current = child
    current.pop(parts[-1], None)
    for parent, part in reversed(parents):
        if parent[part]:
            break
        parent.pop(part)


def flatten_mapping(mapping: dict, prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, path))
        else:
            flattened[path] = value
    return flattened


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def model_paths(model: type[BaseModel], prefix: str = "") -> set[str]:
    paths = set()
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths.update(model_paths(annotation, path))
        else:
            paths.add(path)
    return paths


def model_has_path(model: type[BaseModel], path: str) -> bool:
    return path in model_paths(model)


def filter_for_model(mapping: dict, model: type[BaseModel]) -> dict:
    filtered = {}
    for path, value in flatten_mapping(mapping).items():
        if model_has_path(model, path):
            nested_set(filtered, path, value)
    return filtered
