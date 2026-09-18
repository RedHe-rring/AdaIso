#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
colorize_csv.py

Read a CSV file, choose one numeric column as the scalar field,
normalize it with percentile clipping, map it to a heat palette,
and write a new CSV containing RGB colors.

Default palette matches the original C++ implementation:
    blue -> cyan -> green -> yellow -> red

Example:
    python colorize_csv.py vertices.csv --value-column curvature
    python colorize_csv.py vertices.csv --value-column target_h --invert
    python colorize_csv.py vertices.csv --list-columns
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


RGB = Tuple[int, int, int]

# Matches the original C++ heat_color() palette.
HEAT_PALETTE: Sequence[RGB] = (
    (45, 70, 180),
    (0, 170, 210),
    (80, 190, 80),
    (245, 210, 70),
    (210, 40, 40),
)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, matching the original C++ logic."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    q = clamp(q, 0.0, 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    t = pos - lo
    return sorted_values[lo] * (1.0 - t) + sorted_values[hi] * t


def scalar_range(
    values: Iterable[float],
    clip_low_percentile: float = 2.0,
    clip_high_percentile: float = 98.0,
) -> Tuple[float, float]:
    """
    Compute visualization range from finite values.

    Percentiles are expressed in [0, 100].
    """
    finite = sorted(v for v in values if math.isfinite(v))

    if not finite:
        return 0.0, 1.0

    low_q = clamp(clip_low_percentile / 100.0, 0.0, 1.0)
    high_q = clamp(clip_high_percentile / 100.0, 0.0, 1.0)

    if high_q < low_q:
        raise ValueError("clip_high_percentile must be >= clip_low_percentile")

    lo = quantile(finite, low_q)
    hi = quantile(finite, high_q)

    if hi <= lo:
        pad = max(1.0, abs(lo)) * 0.5
        lo -= pad
        hi += pad

    return lo, hi


def color_byte(value: float) -> int:
    return int(round(clamp(value, 0.0, 1.0) * 255.0))


def lerp_color(a: RGB, b: RGB, t: float) -> RGB:
    t = clamp(t, 0.0, 1.0)
    return (
        color_byte(((1.0 - t) * a[0] + t * b[0]) / 255.0),
        color_byte(((1.0 - t) * a[1] + t * b[1]) / 255.0),
        color_byte(((1.0 - t) * a[2] + t * b[2]) / 255.0),
    )


def heat_color(t: float, palette: Sequence[RGB] = HEAT_PALETTE) -> RGB:
    """Map normalized t in [0, 1] to the heat palette."""
    t = clamp(t, 0.0, 1.0)

    scaled = t * (len(palette) - 1)
    index = min(int(math.floor(scaled)), len(palette) - 2)
    local_t = scaled - index

    return lerp_color(palette[index], palette[index + 1], local_t)


def parse_float(text: str) -> float:
    try:
        value = float(text)
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError):
        return math.nan


def default_output_path(input_path: Path, value_column: str) -> Path:
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in value_column
    )
    return input_path.with_name(
        f"{input_path.stem}_{safe_name}_colored{input_path.suffix}"
    )


def read_csv(input_path: Path):
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        rows = list(reader)
        return reader.fieldnames, rows


def colorize_csv(
    input_path: Path,
    output_path: Path,
    value_column: str,
    clip_low_percentile: float,
    clip_high_percentile: float,
    invert: bool,
    normalized_column: str = "color_t",
    red_column: str = "color_r",
    green_column: str = "color_g",
    blue_column: str = "color_b",
) -> Tuple[float, float, int]:
    fieldnames, rows = read_csv(input_path)

    if value_column not in fieldnames:
        raise KeyError(
            f'Column "{value_column}" not found. '
            f"Available columns: {', '.join(fieldnames)}"
        )

    values: List[float] = [
        parse_float(row.get(value_column, ""))
        for row in rows
    ]

    finite_values = [v for v in values if math.isfinite(v)]
    if not finite_values:
        raise ValueError(
            f'Column "{value_column}" contains no finite numeric values.'
        )

    lo, hi = scalar_range(
        finite_values,
        clip_low_percentile=clip_low_percentile,
        clip_high_percentile=clip_high_percentile,
    )

    out_fieldnames = list(fieldnames)
    for name in (
        normalized_column,
        red_column,
        green_column,
        blue_column,
    ):
        if name not in out_fieldnames:
            out_fieldnames.append(name)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        for row, value in zip(rows, values):
            out_row = dict(row)

            if math.isfinite(value):
                t = clamp((value - lo) / (hi - lo), 0.0, 1.0)
                if invert:
                    t = 1.0 - t

                r, g, b = heat_color(t)
                out_row[normalized_column] = f"{t:.17g}"
                out_row[red_column] = str(r)
                out_row[green_column] = str(g)
                out_row[blue_column] = str(b)
            else:
                # Same conceptual fallback as the C++ visualizer:
                # invalid value is mapped to the low end.
                t = 1.0 if invert else 0.0
                r, g, b = heat_color(t)
                out_row[normalized_column] = ""
                out_row[red_column] = str(r)
                out_row[green_column] = str(g)
                out_row[blue_column] = str(b)

            writer.writerow(out_row)

    return lo, hi, len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Colorize one numeric field of a CSV using a heat palette "
            "and write RGB values into a new CSV."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file.",
    )
    parser.add_argument(
        "--value-column",
        "-c",
        help="Column used as the scalar field, e.g. curvature or target_h.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV path. Default: <input>_<column>_colored.csv",
    )
    parser.add_argument(
        "--clip-low",
        type=float,
        default=2.0,
        help="Low percentile for clipping. Default: 2.",
    )
    parser.add_argument(
        "--clip-high",
        type=float,
        default=98.0,
        help="High percentile for clipping. Default: 98.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Reverse the color direction after normalization.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print available CSV columns and exit.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_csv.exists():
        parser.error(f"Input file does not exist: {args.input_csv}")

    fieldnames, _ = read_csv(args.input_csv)

    if args.list_columns:
        print("Available columns:")
        for name in fieldnames:
            print(f"  {name}")
        return

    if not args.value_column:
        parser.error(
            "--value-column is required unless --list-columns is used."
        )

    if not (0.0 <= args.clip_low <= 100.0):
        parser.error("--clip-low must be between 0 and 100.")
    if not (0.0 <= args.clip_high <= 100.0):
        parser.error("--clip-high must be between 0 and 100.")
    if args.clip_high < args.clip_low:
        parser.error("--clip-high must be >= --clip-low.")

    output_path = (
        args.output
        if args.output is not None
        else default_output_path(args.input_csv, args.value_column)
    )

    lo, hi, row_count = colorize_csv(
        input_path=args.input_csv,
        output_path=output_path,
        value_column=args.value_column,
        clip_low_percentile=args.clip_low,
        clip_high_percentile=args.clip_high,
        invert=args.invert,
    )

    print(f"Input:        {args.input_csv}")
    print(f"Field:        {args.value_column}")
    print(f"Rows:         {row_count}")
    print(f"Color range:  [{lo:.17g}, {hi:.17g}]")
    print(f"Inverted:     {args.invert}")
    print(f"Output:       {output_path}")


if __name__ == "__main__":
    main()
