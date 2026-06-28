#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path("/root/autodl-tmp/ipfp_repro/results/semantic_kitti_repro")
    summaries = sorted(root.glob("*/tiny_overfit_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not summaries:
        raise SystemExit("NO_TINY_OVERFIT_SUMMARY")
    summary = summaries[0]
    leaf = summary.parent
    run = leaf.parent
    for link, target in [(root / "LATEST", run), (root / "LATEST_TINY_OVERFIT", leaf)]:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    print("summary", summary)
    print("leaf", leaf)
    print("run", run)
    print("latest", (root / "LATEST").resolve())
    print("latest_tiny", (root / "LATEST_TINY_OVERFIT").resolve())
    print("files")
    for path in sorted(leaf.iterdir()):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
