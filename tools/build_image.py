"""Build an engine image with source identity. Does not publish or provision anything."""

import argparse
import shlex
import subprocess
from pathlib import Path


def command(root: Path, tag: str, platform: str = "linux/amd64"):
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    )
    return [
        "docker",
        "build",
        "--platform",
        platform,
        "--build-arg",
        f"RELAY_REVISION={revision}",
        "--build-arg",
        f"RELAY_SOURCE_DIRTY={str(dirty).lower()}",
        "-t",
        tag,
        str(root),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--platform", default="linux/amd64")
    args = parser.parse_args()
    argv = command(Path(__file__).resolve().parents[1], args.tag, args.platform)
    print(shlex.join(argv), flush=True)
    if not args.print_only:
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
