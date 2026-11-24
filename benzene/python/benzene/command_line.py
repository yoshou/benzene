"""Command-line pieces the modules share."""

from __future__ import annotations

import argparse


def parse_indices(values: list[str]) -> list[int]:
    """Indices given as numbers and inclusive ranges, separated by spaces or commas: 0 1 2, 0-7, 0-7,96,128."""
    indices = []
    for value in values:
        for part in value.split(","):
            if part:
                first, _, last = part.partition("-")
                indices += range(int(first), int(last or first) + 1)
    return indices


class Indices(argparse.Action):
    """An option of indices (parse_indices): add_argument("--pairs", nargs="+", action=Indices, default=[0])."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, parse_indices(values))
