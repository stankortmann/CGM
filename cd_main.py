# main_cd.py

import argparse
import yaml
from spec_analysis import data_structure as ds
from spec_analysis.column_density.galaxy_swift import run_halo_column_density
from spec_analysis.column_density.box_swift import run_box_column_density  
#for now not importing the box_delta.py, this is not needed for now


def print_cfg(cfg, indent=0):
    """Recursively print all config parameters and values with indentation."""
    prefix = "    " * indent
    for attr in dir(cfg):
        if attr.startswith("_") or callable(getattr(cfg, attr)):
            continue
        value = getattr(cfg, attr)
        if hasattr(value, "__dict__"):
            print(f"{prefix}{attr}:")
            print_cfg(value, indent=indent+1)
        else:
            print(f"{prefix}{attr}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Run column density analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/test.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # --- Load YAML ---
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # --- Create config object ---
    cfg = ds.Config(
        simulation=ds.Simulation(**cfg_dict['simulation']),
        data_output=ds.Data_output(**cfg_dict['data_output']),
        monitoring=ds.Monitoring(**cfg_dict['monitoring']),
        window=ds.Window(**cfg_dict['window']),
        chemistry=ds.Chemistry(**cfg_dict['chemistry']),
        galaxy=ds.Galaxy(**cfg_dict['galaxy'])
    )

    # --- Print all config parameters ---
    print("Configuration parameters and values:")
    print_cfg(cfg)

    # --- Decide which function to call ---
    if getattr(cfg.galaxy, "single_galaxy", False):
        print("\nRunning single-galaxy column density analysis...")
        run_halo_column_density(cfg)
    else:
        print("\nRunning full box column density analysis...")
        run_box_column_density(cfg)


if __name__ == "__main__":
    main()