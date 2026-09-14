"""Top-level command-line interface for RigFL."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

_TEMPLATE_FILES = (
    Path("configs/datasets.yaml"),
    Path("experiments/cifar10_run.yaml"),
    Path("experiments/cifar10_sweep.yaml"),
    Path(".gitignore"),
)


def init_project(directory: str | Path) -> list[Path]:
    """Create a minimal editable RigFL project without overwriting files."""
    destination = Path(directory)
    targets = [destination / relative for relative in _TEMPLATE_FILES]
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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the RigFL command-line interface."""
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
        print("  python -m rigfl.data.generate --dataset cifar10")
        print(
            "  python -m rigfl.experiment.run --algorithm fedavg "
            "--config experiments/cifar10_run.yaml"
        )


if __name__ == "__main__":
    main()
