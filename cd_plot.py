# cd_plot.py

import argparse
import yaml
from pathlib import Path
from spec_analysis.data_structure.plot import plot_config
from spec_analysis.column_density import replot


def load_cfg_plot(cfg_path: str) -> plot_config:
    """
    Load the plotting configuration from a YAML file into a plot_config object.
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Plot configuration file does not exist: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_plot = plot_config(
        label_criterion=cfg_dict["label_criterion"],
        data_directory=cfg_dict["data_directory"],
        length_unit=cfg_dict["length_unit"],
        cd_log_range=cfg_dict["cd_log_range"],
        output_directory=cfg_dict["output_directory"],
        hdf5_files=cfg_dict.get("hdf5_files", None),
        plot_eagle=cfg_dict.get("plot_eagle", False),
        eagle_cddf_directory=Path(cfg_dict.get("eagle_cddf_directory", None)) if cfg_dict.get("eagle_cddf_directory", None) else None
    )

    cfg_plot.validate_paths()
    return cfg_plot

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
    cfg_plot = load_cfg_plot(args.config)

    # --- Get list of HDF5 files ---
    data_files = cfg_plot.data_files

    # --- Decide whether to run single or multiple ---
    if len(data_files) == 1:
        print(f"Replotting single HDF5: {data_files[0]}")
        replot.run_single(cfg_plot)
    else:
        print(f"Replotting multiple HDF5 files: {[str(f) for f in data_files]}")
        replot.run_multiple(cfg_plot)


if __name__ == "__main__":
    main()