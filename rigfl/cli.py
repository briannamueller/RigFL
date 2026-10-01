"""Top-level command-line interface for RigFL."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path

_TEMPLATE_FILES = (
    Path("configs/datasets.yaml"),
    Path("configs/reporting.yaml"),
    Path("configs/experiments/mnist_run.yaml"),
    Path("configs/experiments/mnist_sweep.yaml"),
    Path("configs/experiments/mnist_hpo.yaml"),
    Path(".gitignore"),
)


def init_project(directory: str | Path) -> list[Path]:
    """Create a minimal editable RigFL project without overwriting files."""
    destination = Path(directory)
    relatives = _TEMPLATE_FILES
    targets = [destination / relative for relative in relatives]
    conflicts = [target for target in targets if target.exists()]
    if conflicts:
        listed = "\n".join(f"  {path}" for path in conflicts)
        raise FileExistsError(
            "refusing to overwrite existing scaffold file(s):\n" + listed
        )

    template_root = files("rigfl").joinpath("templates")
    contents = {
        relative: template_root.joinpath(*relative.parts).read_bytes()
        for relative in _TEMPLATE_FILES
    }
    for relative, content in contents.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    return targets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rigfl")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="create a minimal editable RigFL project"
    )
    init.add_argument("directory", help="directory to create")
    task = commands.add_parser(
        "task", help="run one indexed task from a grid or immutable snapshot"
    )
    task.add_argument("grid", help="grid.jsonl or immutable snapshot")
    task.add_argument("index", type=int, help="1-based task index")
    task.add_argument(
        "--results-root",
        help="base results directory; overrides the grid or snapshot setting",
    )
    task.add_argument("--force", action="store_true")
    commands.add_parser(
        "data", add_help=False, help="prepare configured client datasets"
    )
    commands.add_parser(
        "run", add_help=False, help="run one algorithm from a YAML configuration"
    )
    commands.add_parser(
        "sweep", add_help=False, help="expand a YAML sweep into an indexed task grid"
    )
    commands.add_parser(
        "hpo", add_help=False, help="run or resume a YAML-defined Optuna study"
    )
    commands.add_parser(
        "report", add_help=False, help="summarize completed experiment results"
    )
    commands.add_parser(
        "seed-sensitivity",
        add_help=False,
        help="analyze a complete crossed-seed experiment",
    )
    return parser


def _data_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigfl data", description="Prepare configured client datasets."
    )
    commands = parser.add_subparsers(dest="data_command", required=True)
    commands.add_parser("generate", add_help=False, help="generate a client partition")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the RigFL command-line interface."""
    argv = list(sys.argv[1:] if argv is None else argv)

    delegated = {
        "run": ("rigfl.experiment.run", "main"),
        "sweep": ("rigfl.experiment.launch", "main"),
        "hpo": ("rigfl.experiment.optimize", "main"),
        "report": ("rigfl.experiment.collect", "main"),
        "seed-sensitivity": ("rigfl.experiment.variance", "main"),
    }
    if argv and argv[0] in ("run", "sweep", "hpo", "report", "task"):
        from rigfl.experiment.registry import load_plugins

        load_plugins()
    if argv and argv[0] in delegated:
        from importlib import import_module

        module_name, function_name = delegated[argv[0]]
        handler = getattr(import_module(module_name), function_name)
        handler(argv[1:], prog=f"rigfl {argv[0]}")
        return
    if argv and argv[0] == "data":
        if len(argv) >= 2 and argv[1] == "generate":
            from rigfl.data.generate import main as generate

            generate(argv[2:], prog="rigfl data generate")
            return
        _data_parser().parse_args(argv[1:])
        return

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        try:
            created = init_project(args.directory)
        except FileExistsError as error:
            parser.exit(1, f"rigfl init: {error}\n")
        project = Path(args.directory)
        print(f"Created RigFL project in {project}")
        print("\nCreated:")
        for path in created:
            print(f"  {path}")
        print("\nNext:")
        print(f"  cd {project}")
        print("  rigfl data generate --dataset mnist")
        print("  rigfl run configs/experiments/mnist_run.yaml")
        print("  rigfl sweep configs/experiments/mnist_sweep.yaml --execute")
        print("  rigfl report")
    elif args.command == "task":
        from rigfl.experiment.launch import run_task
        run_task(
            args.grid,
            args.index,
            results_root=args.results_root,
            force=args.force,
        )


if __name__ == "__main__":
    main()
