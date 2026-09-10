"""Resolve dataset-declared feature groups for GraphRoute."""

from __future__ import annotations

from functools import partial


def graphroute_feature_extractor(graph, input_spec: dict | None):
    from graphroute.features import BUILTIN_FEATURE_SOURCES, extract_feature_slice

    sources = {graph.node_feature_source, graph.edge_feature_source}
    custom = sorted(sources - BUILTIN_FEATURE_SOURCES)
    if not custom:
        return None
    if len(custom) > 1:
        raise ValueError(
            "FedDES supports one dataset-defined feature source per run; got "
            f"{custom}.")

    source = custom[0]
    input_spec = input_spec or {}
    groups = input_spec.get("feature_groups", {})
    if source not in groups:
        available = ", ".join(sorted(groups)) or "none"
        raise ValueError(
            f"Feature source {source!r} is not available for this dataset; "
            f"available dataset-defined sources: {available}.")

    group = groups[source]
    field_names = [field["name"] for field in input_spec.get("fields", [])]
    field_name = group.get("input")
    if field_name not in field_names:
        raise ValueError(
            f"Feature source {source!r} refers to unknown input {field_name!r}.")
    input_index = field_names.index(field_name)
    start, stop = group.get("start"), group.get("stop")
    width = input_spec["fields"][input_index]["shape"][-1]
    if not isinstance(start, int) or not isinstance(stop, int):
        raise ValueError(
            f"Feature source {source!r} must define integer start and stop values.")
    if not 0 <= start < stop <= width:
        raise ValueError(
            f"Feature source {source!r} has invalid slice [{start}:{stop}] for "
            f"input {field_name!r} with width {width}.")
    return partial(
        extract_feature_slice,
        input_index=input_index,
        start=start,
        stop=stop,
    )
