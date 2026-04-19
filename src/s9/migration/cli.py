"""CLI entry point for checkpoint migration."""

from __future__ import annotations

import argparse
import sys

import torch

from s9.migration.zoh import migrate_state_dict_zoh, migrate_state_dict_from_zoh


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Migrate s9 checkpoints between approx and zoh discretization."
    )
    parser.add_argument("--in", dest="input", required=True, help="Input checkpoint path")
    parser.add_argument("--out", dest="output", required=True, help="Output checkpoint path")
    parser.add_argument(
        "--family", default="auto",
        choices=["auto", "S9", "RS9", "ARS9"],
        help="Kernel family (default: auto-detect)",
    )
    parser.add_argument(
        "--direction", default="to_zoh",
        choices=["to_zoh", "from_zoh"],
        help="Migration direction (default: to_zoh)",
    )
    parser.add_argument(
        "--key", default=None,
        help="Nested key in checkpoint dict for the state_dict (e.g. 'model')",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving")

    args = parser.parse_args(argv)

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)

    if args.key:
        sd = checkpoint[args.key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        args.key = "state_dict"
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        sd = checkpoint["model"]
        args.key = "model"
    else:
        sd = checkpoint
        args.key = None

    migrate_fn = migrate_state_dict_zoh if args.direction == "to_zoh" else migrate_state_dict_from_zoh
    new_sd = migrate_fn(sd, kernel_family=args.family)

    if args.dry_run:
        changed = sum(1 for k in sd if not torch.equal(sd[k], new_sd[k]))
        print(f"Would modify {changed} parameters. Dry run — not saving.")
        return

    if args.key:
        checkpoint[args.key] = new_sd
        torch.save(checkpoint, args.output)
    else:
        torch.save(new_sd, args.output)

    print(f"Migrated checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
