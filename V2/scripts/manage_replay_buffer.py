from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def bytes_in_tree(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage replay buffer storage for V2 experiments.")
    parser.add_argument("--root", default="V2/replay_buffers/raw")
    parser.add_argument("--algo", default="dqn")
    parser.add_argument("--task", default="foundation_env")
    parser.add_argument("--freshness", default="fresh")
    parser.add_argument("--max-gb", type=float, default=1.0)
    parser.add_argument("--keep-latest", type=int, default=10)
    parser.add_argument("--replace-from", default="")
    parser.add_argument("--replace-to", default="")
    args = parser.parse_args()

    root = Path(args.root)
    target = root / args.algo / args.task / args.freshness
    target.mkdir(parents=True, exist_ok=True)

    total_bytes = bytes_in_tree(root)
    max_bytes = int(args.max_gb * (1024**3))
    assert total_bytes <= max_bytes, (
        f"Replay buffer tree exceeds limit: {total_bytes} bytes > {max_bytes} bytes. "
        "Raise --max-gb or remove older experience."
    )

    files = sorted([p for p in target.glob("*.npz") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    for old_file in files[args.keep_latest:]:
        old_file.unlink()

    if args.replace_from and args.replace_to:
        src = Path(args.replace_from)
        dst = Path(args.replace_to)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied newer replay data from {src} to {dst}")

    final_bytes = bytes_in_tree(root)
    print(f"Replay buffer root: {root.resolve()}")
    print(f"Total size: {final_bytes} bytes")
    print(f"Target directory: {target.resolve()}")


if __name__ == "__main__":
    main()
