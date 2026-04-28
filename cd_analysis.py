# main_cd.py

import argparse
import yaml
import cddf.data_structure.simulation as ds
from cddf.pipelines.galaxy_swift import run_halo_column_density
from cddf.pipelines.box_swift import run_box_column_density  
from cddf.pipelines.box_mpi import run_box_column_density_parallel
from cddf.pipelines.box_mpi_multiple import run_slice_column_density_parallel
from pathlib import Path
import h5py
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
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size == 1:
            comm = None
    except ImportError:
        comm = None
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
        cddf=ds.Cddf(**cfg_dict['cddf']),
        galaxy=ds.Galaxy(**cfg_dict['galaxy']),
        omega_ion=ds.Omega_ion(**cfg_dict['omega_ion'])
    )

    # --- Print all config parameters ---
    if rank == 0:
        print("Configuration parameters and values:")
        print_cfg(cfg)

    # --- Decide which function to call ---
    if getattr(cfg.galaxy, "single_galaxy", False):
        print("\nRunning single-galaxy column density analysis...") if rank == 0 else None
        run_halo_column_density(cfg)
    else:
        print("\nRunning full box column density analysis...") if rank == 0 else None
        if comm is not None and size > 1:
            if cfg.window.projection_slices ==1:
                print(f"Running full box in parallel with {size} cores.") if rank == 0 else None
                run_box_column_density_parallel(cfg, comm)
            elif cfg.window.projection_slices > 1:
                print(f"Running {cfg.window.projection_slices} slice-based column density in parallel with {size} cores.") if rank == 0 else None
                run_slice_column_density_parallel(cfg, comm)
        else:
            print(f"Running on a single core.")
            run_box_column_density(cfg)


if __name__ == "__main__":
    main()