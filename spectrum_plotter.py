import argparse
import os
from spectra.long_spectra import long_spectra
from spectra.short_spectra import short_spectra


def main():
    parser = argparse.ArgumentParser(description="Plot all short, long, or shortquery spectra from a SpecWizard HDF5 output or directory of LOS outputs.")
    parser.add_argument("--file", required=False, help="Path to optical-depth HDF5 output")
    parser.add_argument("--dir", required=False, help="Path to directory containing short spectra HDF5 files for shortquery mode")
    parser.add_argument("--mode", required=True, choices=["short", "long", "shortquery"], default="short", help="Which plot style to use")

    args = parser.parse_args()

    if args.mode == "long":
        spectra = long_spectra(wizard={"Output": {"directory": ".", "fname": args.file}})
        spectra.plot_spectra(hdf5_file=args.file, show=True)
    elif args.mode == "short":
        spectra = short_spectra(wizard={"Output": {"directory": ".", "fname": args.file}})
        spectra.plot_spectra(hdf5_file=args.file, show=True)
    elif args.mode == "shortquery":
        if not args.dir:
            raise ValueError("--dir must be specified for shortquery mode")
        spectra = short_spectra(wizard={"Output": {"directory": args.dir, "fname": "."}})
        spectra.inspect_los(args.dir)

if __name__ == "__main__":
    main()
