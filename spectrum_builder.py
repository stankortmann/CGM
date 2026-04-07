import argparse
import copy

import specwizard


from .spectra.short_spectra import short_spectra


DEFAULT_YAML = "/cosma/home/do012/dc-kort1/CGM/configurations/specwizard/template/spec.yaml"


def main():
    parser = argparse.ArgumentParser(description="Build spectra for a range of sightlines")
    parser.add_argument(
        "--yml-file",
        default=DEFAULT_YAML,
        help="Path to the specwizard YAML file",
    )
    parser.add_argument(
        "--nsight-range",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        required=True,
        help="Inclusive range of sightline numbers to run",
    )
    args = parser.parse_args()

    if args.nsight_range[1] < args.nsight_range[0]:
        raise ValueError("nsight-range stop must be greater than or equal to start")

    build_input = specwizard.Build_Input()
    base_wizard = build_input.read_from_yml(yml_file=args.yml_file)

    for nsight in range(args.nsight_range[0], args.nsight_range[1] + 1):
        wizard = copy.deepcopy(base_wizard)
        wizard.setdefault("sightline", {})["nsight"] = nsight
        print(f"Running sightline {nsight}")
        short_spectra(wizard).run()


if __name__ == "__main__":
    main()
