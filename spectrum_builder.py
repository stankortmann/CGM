import argparse
import copy

import specwizard

from spectra.long_spectra import long_spectra
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


class spectrum_builder:
    def __init__(self, config_path):
        self.config_path = config_path
        self._wizard = None

    @property
    def wizard(self):
        if self._wizard is None:
            build_input = specwizard.Build_Input()
            self._wizard = build_input.read_from_yml(yml_file=self.config_path)
        return self._wizard

    def run_short_range(self, start, stop):
        results = []
        for nsight in range(start, stop + 1):
            wizard = copy.deepcopy(self.wizard)
            wizard.setdefault("sightline", {})["nsight"] = nsight
            print(f"Running short spectra for sightline {nsight}")
            result = short_spectra(wizard).run_spectra()
            results.append(result)
            if result is not None and "hdf5_file" in result:
                print(f"Short spectra HDF5: {result['hdf5_file']}")
        return results

    def run_long(self):
        print("Running long spectra")
        runner = long_spectra(copy.deepcopy(self.wizard))
        result = runner.run_spectra()
        hdf5_file = result.get("hdf5_file") if isinstance(result, dict) else None
        if hdf5_file is not None:
            print(f"Long spectra HDF5: {hdf5_file}")
        return result


def main():
    parser = argparse.ArgumentParser(description="Build short or long spectra")

    parser.add_argument(
        "--mode",
        choices=["short", "long", "plot-hdf5"],
        default="short",
        help="Which spectra pipeline to run",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_YAML,
        help="Path to the specwizard YAML file",
    )

    
    parser.add_argument(
        "--nsight-range",
        nargs="+",
        metavar=("START", "STOP"),
        help="Inclusive short-spectra sightline range; use 'START STOP' or 'START,STOP'",
    )

    args = parser.parse_args()

    builder = spectrum_builder(config_path=args.config)

    if args.mode == "short":
        if args.nsight_range is None:
            parser.error("--nsight-range is required in --mode short")
        start, stop = parse_nsight_range(args.nsight_range)
        builder.run_short_range(start, stop)
        return

    if args.mode == "long":
        builder.run_long()
        return

    parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
