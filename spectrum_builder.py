import argparse
import copy
import os
import sys

import specwizard

from spectra.short_spectra import short_spectra



DEFAULT_YAML = "/cosma/home/do012/dc-kort1/CGM/configurations/spectra/short/L25/z0.yaml"


def parse_nsight_range(values):
    if len(values) == 1:
        parts = values[0].split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("nsight-range must be START STOP or START,STOP")
        try:
            start, stop = (int(part.strip()) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("nsight-range values must be integers") from exc
    elif len(values) == 2:
        start, stop = values
    else:
        raise argparse.ArgumentTypeError("nsight-range must be START STOP or START,STOP")

    if stop < start:
        raise argparse.ArgumentTypeError("nsight-range stop must be greater than or equal to start")

    return start, stop


def main():
    parser = argparse.ArgumentParser(description="Build spectra for a range of sightlines")
    parser.add_argument(
        "--config",
        default=DEFAULT_YAML,
        help="Path to the specwizard YAML file",
    )
    parser.add_argument(
        "--nsight-range",
        nargs="+",
        required=True,
        metavar=("START", "STOP"),
        help="Inclusive range of sightline numbers to run; use 'START STOP' or 'START,STOP'",
    )
    args = parser.parse_args()

    start, stop = parse_nsight_range(args.nsight_range)

    build_input = specwizard.Build_Input()
    base_wizard = build_input.read_from_yml(yml_file=args.config)

    for nsight in range(start, stop + 1):
        wizard = copy.deepcopy(base_wizard)
        wizard.setdefault("sightline", {})["nsight"] = nsight
        print(f"Running sightline {nsight}")
        short_spectra(wizard).run()


if __name__ == "__main__":
    main()
