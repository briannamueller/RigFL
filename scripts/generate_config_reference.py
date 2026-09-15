"""Generate Markdown references from RigFL's configuration models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Annotated, Literal, get_args, get_origin

from graphroute.config import GraphRouteSettings
from pydantic import BaseModel

from rigfl.data.config import (
    BioSiloDatasetSettings,
    ClientSplitSettings,
    FlowerDatasetSettings,
    MergedSourceSplits,
    PartitionSettings,
    PartitionSettingsBase,
    SourceSplits,
)
from rigfl.experiment.config import (
    CategoricalDistribution,
    EarlyStoppingConfig,
    ExperimentConfig,
    ExperimentFileConfig,
    FloatDistribution,
    IntegerDistribution,
    IntensificationConfig,
    ReplicateCondition,
    SamplerConfig,
    TuningConfig,
)
from rigfl.experiment.registry import REGISTRY
from rigfl.models.registry import MODEL_ARCHITECTURE_REGISTRY, MODEL_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "docs" / "reference"
_MISSING = object()
_FRIENDLY_TYPES = {
    "BaseConfiguration": "mapping",
    "SweepConfiguration": "mapping",
    "ReplicateCondition": "replicate settings",
    "TuningConfig": "tuning settings",
    "SourceSplits": "named source splits",
    "MergedSourceSplits": "merged source splits",
    "ClientSplitSettings": "client split settings",
}


def _variants(annotation) -> list[type[BaseModel]]:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return list(get_args(annotation))


def _resolve(schema: dict, root: dict) -> dict:
    if "$ref" in schema:
        target = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        return target
    return schema


def _positive_type(schema: dict) -> str | None:
    kind = schema.get("type")
    if kind == "integer" and schema.get("minimum") == 1:
        return "positive integers"
    if kind == "number" and schema.get("exclusiveMinimum") == 0:
        return "positive numbers"
    return None


def _type_name(schema: dict, root: dict) -> str:
    if "$ref" in schema:
        title = _resolve(schema, root).get("title", "object")
        return _FRIENDLY_TYPES.get(title, title)
    if "anyOf" in schema:
        names = []
        for item in schema["anyOf"]:
            name = _type_name(item, root)
            if name not in names:
                names.append(name)
        if "number" in names and "integer" in names:
            names.remove("integer")
        return " | ".join(names)
    kind = schema.get("type")
    if kind == "array":
        item_schema = schema.get("items", {})
        items = _resolve(item_schema, root)
        positive = _positive_type(items)
        if positive:
            return f"list of {positive}"
        return f"list[{_type_name(item_schema, root)}]"
    if kind == "object":
        values = schema.get("additionalProperties")
        if isinstance(values, dict):
            return f"mapping[string, {_type_name(values, root)}]"
        return "mapping"
    return {
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "string": "string",
        "null": "null",
    }.get(kind, "value")


def _code(value) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return f"`{text.replace('|', '&#124;')}`"


def _constraints(schema: dict) -> str:
    parts = []
    enums = schema.get("enum")
    if enums is not None:
        parts.append(", ".join(_code(value) for value in enums))
    if "const" in schema:
        parts.append(_code(schema["const"]))
    numeric_bounds = (
        ("minimum", "≥"),
        ("exclusiveMinimum", ">"),
        ("maximum", "≤"),
        ("exclusiveMaximum", "<"),
        ("ge", "≥"),
        ("gt", ">"),
        ("le", "≤"),
        ("lt", "<"),
    )
    for key, label in numeric_bounds:
        if key in schema:
            parts.append(f"{label} {schema[key]}")
    if "minLength" in schema:
        length = schema["minLength"]
        parts.append("non-empty" if length == 1 else f"at least {length} characters")
    if "maxLength" in schema:
        parts.append(f"at most {schema['maxLength']} characters")
    if "minItems" in schema:
        count = schema["minItems"]
        parts.append("non-empty" if count == 1 else f"at least {count} items")
    if "maxItems" in schema:
        count = schema["maxItems"]
        parts.append(f"at most {count} item" + ("" if count == 1 else "s"))
    items = schema.get("items")
    if isinstance(items, dict):
        item_constraints = _constraints(items)
        if item_constraints != "—":
            constraints = item_constraints.split("; ")
            positive = _positive_type(items)
            if positive == "positive integers":
                constraints = [value for value in constraints if value != "≥ 1"]
            elif positive == "positive numbers":
                constraints = [value for value in constraints if value != "> 0"]
            if constraints:
                parts.append("each item " + "; ".join(constraints))
    if "anyOf" in schema:
        choices = []
        for item in schema["anyOf"]:
            value = _constraints(item)
            if value != "—" and value not in choices:
                choices.append(value)
        if choices:
            parts.append(" or ".join(choices))
    return "; ".join(parts) or "—"


def _model_defaults(model: type[BaseModel]) -> dict:
    defaults = {}
    for name, field in model.model_fields.items():
        if field.is_required():
            continue
        value = field.get_default(call_default_factory=True)
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", by_alias=True)
        defaults[field.alias or name] = value
    return defaults


def _rows(
    model: type[BaseModel],
    prefix: str = "",
    *,
    exclude: set[str] | None = None,
    defaults: dict | None = None,
) -> list[list[str]]:
    root = model.model_json_schema(by_alias=True)
    exclude = exclude or set()
    defaults = _model_defaults(model) if defaults is None else defaults
    rows: list[list[str]] = []

    def visit(node: dict, path: str, values=_MISSING, required: bool = False):
        resolved = _resolve(node, root)
        properties = resolved.get("properties")
        if properties:
            required_names = set(resolved.get("required", []))
            current = values if isinstance(values, dict) else {}
            for name, child in properties.items():
                if path == prefix and name in exclude:
                    continue
                child_values = current.get(name, _MISSING)
                visit(
                    child,
                    f"{path}.{name}" if path else name,
                    child_values,
                    name in required_names,
                )
            return

        default = values
        if default is _MISSING:
            default = resolved.get("default", _MISSING)
        default_text = "required" if required and default is _MISSING else "—"
        if default is not _MISSING:
            default_text = _code(default)
        description = resolved.get("description", "—").replace("\n", " ")
        rows.append(
            [
                f"`{path}`",
                _type_name(node, root).replace("|", "\\|"),
                default_text,
                _constraints(resolved),
                description.replace("|", "\\|"),
            ]
        )

    visit(root, prefix, defaults)
    return rows


def _table(rows: list[list[str]]) -> str:
    lines = [
        "| Setting | Type | Default | Allowed | Description |",
        "|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _defaults_table(rows: list[list[str]], upstream_rows: dict[str, list[str]]) -> str:
    lines = [
        "| Setting | GraphRoute default | RigFL default |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {row[0]} | {upstream_rows[row[0]][2]} | {row[2]} |" for row in rows
    )
    return "\n".join(lines) + "\n"


def _section(
    title: str,
    model: type[BaseModel],
    prefix: str = "",
    *,
    exclude: set[str] | None = None,
    defaults: dict | None = None,
    level: int = 2,
) -> str:
    return (
        f"{'#' * level} {title}\n\n"
        f"{_table(_rows(model, prefix, exclude=exclude, defaults=defaults))}"
    )


def _header(title: str, intro: str | None = None) -> str:
    text = f"# {title}\n"
    return f"{text}\n{intro}\n" if intro else text


def _changed_paths(left: dict, right: dict, prefix: str = "") -> set[str]:
    changed = set()
    for name in left.keys() | right.keys():
        path = f"{prefix}.{name}" if prefix else name
        left_value = left.get(name, _MISSING)
        right_value = right.get(name, _MISSING)
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            changed |= _changed_paths(left_value, right_value, path)
        elif left_value != right_value:
            changed.add(path)
    return changed


def render_experiment_reference() -> str:
    parts = [
        _header(
            "Experiment configuration reference",
            "These options configure individual runs, sweeps, and tuning studies.",
        ),
        _section("Experiment settings", ExperimentConfig, "experiment"),
        _section("Sweep and study files", ExperimentFileConfig),
        _section("Replicate conditions", ReplicateCondition, "replicates[]")
        + (
            "\nReplicate entries must be unique and use distinct "
            "`experiment_seed` values. Replicate seed fields cannot also be "
            "sweep axes.\n"
        ),
        _section(
            "Tuning settings",
            TuningConfig,
            "tuning",
            exclude={"sampler", "intensification"},
        ),
        _section(
            "Sampler settings",
            SamplerConfig,
            "tuning.sampler",
            defaults=_model_defaults(TuningConfig)["sampler"],
        ),
        _section(
            "Intensification settings", IntensificationConfig, "tuning.intensification"
        ),
        _section(
            "Categorical search parameters",
            CategoricalDistribution,
            "tuning.search_space.<name>",
        ),
        _section(
            "Integer search parameters",
            IntegerDistribution,
            "tuning.search_space.<name>",
        ),
        _section(
            "Floating-point search parameters",
            FloatDistribution,
            "tuning.search_space.<name>",
        )
        + (
            "\n`log: true` cannot be combined with `step`. Integer ranges "
            "allow equal bounds; floating-point ranges require `low < high`.\n"
        ),
    ]
    return "\n".join(parts)


def render_algorithm_reference() -> str:
    rows = []
    for name, spec in REGISTRY.items():
        rows.append(
            [
                f"`{name}`",
                "yes" if spec.supports_model_heterogeneity else "no",
            ]
        )
    capability_table = (
        "\n".join(
            [
                "| Algorithm | Heterogeneous models |",
                "|---|---|",
                *("| " + " | ".join(row) + " |" for row in rows),
            ]
        )
        + "\n"
    )
    parts = [
        _header(
            "Algorithm configuration reference",
        ),
        "## Capabilities\n\n" + capability_table,
    ]
    for name, spec in REGISTRY.items():
        if name == "feddes":
            effective = spec.config().graphroute.model_dump(mode="json")
            upstream = GraphRouteSettings().model_dump(mode="json")
            changed = {
                f"`algorithm.graphroute.{path}`"
                for path in _changed_paths(effective, upstream)
            }
            generated_rows = _rows(spec.config, "algorithm")
            upstream_rows = {
                row[0]: row for row in _rows(GraphRouteSettings, "algorithm.graphroute")
            }
            cache_row = next(
                row for row in generated_rows if row[0] == "`algorithm.cache_dir`"
            )
            override_rows = [row for row in generated_rows if row[0] in changed]
            settings_rows = [
                [
                    "`algorithm.graphroute`",
                    "mapping",
                    "see below",
                    "—",
                    "Settings passed to GraphRoute.",
                ],
                cache_row,
            ]
            parts.append(
                f"## `{name}`\n\n"
                "`base.models` is set from the experiment's model selection and "
                "cannot be configured here. `base.split_mode` must be "
                "`oof_stacking`. Other "
                "settings under `algorithm.graphroute` follow the "
                "[GraphRoute configuration guide]"
                "(https://github.com/briannamueller/GraphRoute#configuration).\n\n"
                f"{_table(settings_rows)}\n"
                "### GraphRoute defaults changed by RigFL\n\n"
                f"{_defaults_table(override_rows, upstream_rows)}"
            )
            continue
        parts.append(f"## `{name}`\n\n{_table(_rows(spec.config, 'algorithm'))}")
    return "\n".join(parts)


def render_data_reference() -> str:
    flower_rows = _rows(FlowerDatasetSettings)
    for row in flower_rows:
        if row[0] == "`partition`":
            row[1] = "partition settings"
            row[2] = "`dirichlet`"
    parts = [
        _header(
            "Data configuration reference",
        ),
        "## Flower datasets\n\n" + _table(flower_rows),
        _section("Named source splits", SourceSplits, "source_splits", level=3),
        _section("Merged source splits", MergedSourceSplits, "source_splits", level=3),
        _section("Client splits", ClientSplitSettings, "client_split", level=3)
        + (
            "\n`client_split` and `source_splits.merge_splits` must be used "
            "together. Merged split names must be unique, and the validation "
            "and test fractions must sum to less than 1.\n"
        ),
        _section(
            "Common partition settings",
            PartitionSettingsBase,
            "partition",
            level=3,
        ),
    ]
    common = set(PartitionSettingsBase.model_fields)
    for model in _variants(PartitionSettings):
        scheme = model.model_fields["scheme"].default
        section = _section(
            f"`{scheme}` partitioning",
            model,
            "partition",
            exclude=common,
            level=3,
        )
        if scheme == "dirichlet":
            section += "\nA list-valued `alpha` must contain one value per client.\n"
        elif scheme == "shard":
            section += (
                "\nSet at least one of `num_shards_per_partition` or `shard_size`.\n"
            )
        parts.append(section)
    parts.append(
        _section("BioSilo datasets", BioSiloDatasetSettings)
        + (
            "\n`parameters` cannot set `dataset`, `params`, `root`, `overwrite`, "
            "or `version`; RigFL supplies those arguments.\n"
        )
    )
    return "\n".join(parts)


def render_model_reference() -> str:
    architecture_rows = [
        [f"`{name}`", f"`{input_kind}`"]
        for name, (input_kind, _) in MODEL_ARCHITECTURE_REGISTRY.items()
    ]
    family_rows = [
        [f"`{name}`", ", ".join(f"`{model}`" for model in models)]
        for name, models in MODEL_FAMILIES.items()
    ]
    return "".join(
        [
            _header(
                "Model reference",
            ),
            "\n## Architectures\n\n",
            "| Architecture | Input type |\n|---|---|\n",
            *("| " + " | ".join(row) + " |\n" for row in architecture_rows),
            "\n## Model families\n\n",
            "| Family | Architectures |\n|---|---|\n",
            *("| " + " | ".join(row) + " |\n" for row in family_rows),
        ]
    )


def rendered_documents() -> dict[Path, str]:
    return {
        REFERENCE_DIR / "experiment.md": render_experiment_reference(),
        REFERENCE_DIR / "algorithms.md": render_algorithm_reference(),
        REFERENCE_DIR / "data.md": render_data_reference(),
        REFERENCE_DIR / "models.md": render_model_reference(),
    }


def rigfl_owned_models() -> set[type[BaseModel]]:
    models = {
        EarlyStoppingConfig,
        ExperimentConfig,
        ExperimentFileConfig,
        ReplicateCondition,
        SamplerConfig,
        TuningConfig,
        IntensificationConfig,
        CategoricalDistribution,
        IntegerDistribution,
        FloatDistribution,
        SourceSplits,
        MergedSourceSplits,
        ClientSplitSettings,
        PartitionSettingsBase,
        FlowerDatasetSettings,
        BioSiloDatasetSettings,
        *(_variants(PartitionSettings)),
    }
    models.update(spec.config for spec in REGISTRY.values())
    return models


def missing_descriptions() -> list[str]:
    missing = []
    for model in sorted(rigfl_owned_models(), key=lambda item: item.__name__):
        for name, field in model.model_fields.items():
            annotation = field.annotation
            if get_origin(annotation) is Annotated:
                annotation = get_args(annotation)[0]
            single_literal = (
                get_origin(annotation) is Literal and len(get_args(annotation)) == 1
            )
            if field.description is None and not single_literal:
                missing.append(f"{model.__name__}.{name}")
    return missing


def write_documents(*, check: bool = False) -> bool:
    stale = []
    for path, content in rendered_documents().items():
        content = content.rstrip() + "\n"
        if path.exists() and path.read_text() == content:
            continue
        stale.append(path)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if check and stale:
        print("Configuration reference is stale:")
        for path in stale:
            print(f"  {path.relative_to(ROOT)}")
        print("Run: python scripts/generate_config_reference.py")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if write_documents(check=args.check) else 1)


if __name__ == "__main__":
    main()
