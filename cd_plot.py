# cd_plot.py

import argparse
import yaml
from pathlib import Path
from cddf.data_structure.plot import plot_config
from cddf.pipelines import plotter


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
        load_cd=cfg_dict.get("load_cd", False),
        stack_total_label=cfg_dict.get("stack_total_label", False),
        slice_label=cfg_dict.get("slice_label", False),
        Z_label=cfg_dict.get("Z_label", False),
        plot_eagle=cfg_dict.get("plot_eagle", False),
        eagle_cddf_directory=Path(cfg_dict.get("eagle_cddf_directory", None)) if cfg_dict.get("eagle_cddf_directory", None) else None,
        observational_cddf_directory=Path(cfg_dict.get("observational_cddf_directory", None)) if cfg_dict.get("observational_cddf_directory", None) else None,
        plot_observations=cfg_dict.get("plot_observations", False),
        plot_2d_histogram=cfg_dict.get("plot_2d_histogram", False),
        galaxy_plot=cfg_dict.get("galaxy_plot", False)
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
    if len(data_files) == 1 and not cfg_plot.galaxy_plot:
        print(f"Replotting single HDF5: {data_files[0]}")
        plotter.run_single(cfg_plot)

    elif len(data_files) > 1 and not cfg_plot.galaxy_plot:
        print(f"Replotting multiple HDF5 files: {[str(f) for f in data_files]}")
        plotter.run_multiple(cfg_plot)

    elif len(data_files) == 1 and cfg_plot.galaxy_plot:
        print(f"Replotting single galaxy HDF5: {data_files[0]}")
        plotter.run_single_halo(cfg_plot)
    
    elif len(data_files) > 1 and cfg_plot.galaxy_plot:
        print(f"Replotting multiple galaxy HDF5 files: {[str(f) for f in data_files]}")
        plotter.run_multiple_halos(cfg_plot)
    else:
        raise ValueError("Invalid configuration: check that hdf5_files is set correctly and galaxy_plot flag is consistent.")


if __name__ == "__main__":
    main()