# cd_plot.py

import argparse
import yaml
from pathlib import Path
from spec_analysis.data_structure_plot import plot_config
from spec_analysis.column_density import replot

def load_plot_cfg(cfg_path: str) -> plot_config:
    """
    Load the plotting configuration from a YAML file into a plot_config object.
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Plot configuration file does not exist: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    plot_cfg = plot_config(
        label_criterion=cfg_dict["label_criterion"],
        data_directory=cfg_dict["data_directory"],
        output_directory=cfg_dict["output_directory"],
        hdf5_files=cfg_dict.get("hdf5_files", None)
    )

    plot_cfg.validate_paths()
    return plot_cfg

def main():
    parser = argparse.ArgumentParser(description="Replot column densities from HDF5 files")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/cfg_plot.yaml",
        help="Path to YAML plotting configuration file"
    )
    args = parser.parse_args()

    # --- Load plot_config ---
    plot_cfg = load_plot_cfg(args.config)

    # --- Get list of HDF5 files ---
    hdf5_files = plot_cfg.get_hdf5_files()

    # --- Decide whether to run single or multiple ---
    if len(hdf5_files) == 1:
        print(f"Replotting single HDF5: {hdf5_files[0]}")
        replot.run_single(hdf5_files[0], plot_cfg)
    else:
        print(f"Replotting multiple HDF5 files: {[str(f) for f in hdf5_files]}")
        replot.run_multiple(hdf5_files, plot_cfg)


if __name__ == "__main__":
    main()