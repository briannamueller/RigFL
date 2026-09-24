"""Top-level command-line interface for RigFL."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path

from rigfl import __version__

_TEMPLATE_FILES = (
    Path("configs/datasets.yaml"),
    Path("configs/reporting.yaml"),
    Path("configs/experiments/cifar10_run.yaml"),
    Path("configs/experiments/cifar10_sweep.yaml"),
    Path(".gitignore"),
)
_REQUIREMENTS_FILE = Path("requirements.txt")


def legacy_config_argv(argv: list[str] | None) -> list[str]:
    """Translate the former hidden ``--config FILE`` spelling to ``CONFIG``."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    for index, argument in enumerate(arguments):
        if argument == "--config":
            if index + 1 >= len(arguments):
                return arguments
            config = arguments[index + 1]
            del arguments[index:index + 2]
            return [config, *arguments]
        if argument.startswith("--config="):
            config = argument.split("=", 1)[1]
            del arguments[index]
            return [config, *arguments]
    return arguments


def init_project(directory: str | Path) -> list[Path]:
    """Create a minimal editable RigFL project without overwriting files."""
    destination = Path(directory)
    relatives = (*_TEMPLATE_FILES, _REQUIREMENTS_FILE)
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
    contents[_REQUIREMENTS_FILE] = f"rigfl=={__version__}\n".encode()
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
    snapshot = commands.add_parser(
        "snapshot", help="freeze a task grid for external execution"
    )
    snapshot.add_argument("grid", help="working grid.jsonl to snapshot")
    snapshot.add_argument("--results-root", default="results")
    task = commands.add_parser(
        "task", help="run one indexed task from a grid or immutable snapshot"
    )
    task.add_argument("grid", help="grid.jsonl or immutable snapshot")
    task.add_argument("index", type=int, help="1-based task index")
    task.add_argument("--results-root", default="results")
    task.add_argument("--dry-run", action="store_true")
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
        print("  rigfl data generate --dataset cifar10")
        print(
            "  rigfl run configs/experiments/cifar10_run.yaml "
            "--algorithm fedavg"
        )
    elif args.command == "snapshot":
        from rigfl.experiment.launch import stage_task_snapshot

        path = stage_task_snapshot(Path(args.grid), args.results_root)
        print(f"Wrote immutable task snapshot: {path}")
        print(f"Run task 1 with:\n  rigfl task {path} 1")
    elif args.command == "task":
        from rigfl.experiment.launch import run_task
        from rigfl.experiment.storage import run_store

        run_task(
            args.grid,
            args.index,
            run_store(args.results_root),
            dry_run=args.dry_run,
            force=args.force,
        )


if __name__ == "__main__":
    main()
