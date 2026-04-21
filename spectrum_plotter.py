import argparse

from spectra.long_spectra import long_spectra
from spectra.short_spectra import short_spectra


def main():
    parser = argparse.ArgumentParser(description="Plot all short or long spectra from a SpecWizard HDF5 output")
    parser.add_argument("--file", required=True, help="Path to optical-depth HDF5 output")
    parser.add_argument("--mode", choices=["short", "long"], default="short", help="Which plot style to use")

    args = parser.parse_args()

    if args.mode == "long":
        spectra = long_spectra(wizard={"Output": {"directory": ".", "fname": args.file}})
        spectra.plot_spectra(hdf5_file=args.file, show=True)
    else:
        spectra = short_spectra(wizard={"Output": {"directory": ".", "fname": args.file}})
        spectra.plot_spectra(hdf5_file=args.file, show=True)


if __name__ == "__main__":
    main()
