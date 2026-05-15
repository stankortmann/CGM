# main_cd.py

import argparse
import yaml
import cddf.data_structure.simulation as ds
from cddf.pipelines.galaxy_swift import run_halo_column_density
from cddf.pipelines.box_swift import run_box_column_density  
from cddf.pipelines.box_mpi import run_box_column_density_parallel
from cddf.pipelines.box_mpi_multiple import run_slice_column_density_parallel
from cddf.pipelines.omega_parameter import run_omega_parameter
from pathlib import Path
import h5py
import threading
from time import sleep
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


def _read_rss_bytes():
    """Read resident set size for the current process from /proc/self/status."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        pass
    return 0


def monitor_system(cfg, comm, rank_id=0):
    """Print combined RSS across all ranks every cfg.monitoring.monitor_interval seconds."""
    if not getattr(cfg.monitoring, "cpu_ram_monitor", False):
        return

    interval = int(getattr(cfg.monitoring, "monitor_interval", 100))
    if interval < 1:
        interval = 1

    from mpi4py import MPI

    while True:
        local_rss = _read_rss_bytes()
        total_rss = comm.allreduce(local_rss, op=MPI.SUM) if comm is not None else local_rss
        if rank_id == 0:
            total_gb = total_rss / (1024 ** 3)
            print(f"[SYSTEM MONITOR] combined RSS across all ranks: {total_gb:.2f} GB")
        sleep(interval)


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
        simulation=ds.Simulation(**cfg_dict.get('simulation', {})),
        data_output=ds.Data_output(**cfg_dict.get('data_output', {})),
        monitoring=ds.Monitoring(**cfg_dict.get('monitoring', {})),
        window=ds.Window(**cfg_dict.get('window', {})),
        chemistry=ds.Chemistry(**cfg_dict.get('chemistry', {})),
        cddf=ds.Cddf(**cfg_dict.get('cddf', {})),
        galaxy=ds.Galaxy(**cfg_dict.get('galaxy', {})),
        omega_ion=ds.Omega_ion(**cfg_dict.get('omega_ion', {}))
    )

    # --- Print all config parameters ---
    if rank == 0:
        print("Configuration parameters and values:")
        print_cfg(cfg)

    monitor_thread = None
    if getattr(cfg.monitoring, "cpu_ram_monitor", False):
        monitor_thread = threading.Thread(
            target=monitor_system,
            kwargs={"cfg": cfg, "comm": comm, "rank_id": rank},
            daemon=True,
        )
        monitor_thread.start()

    # --- Decide which function to call ---
    if getattr(cfg.galaxy, "single_galaxy", False):
        print("\nRunning single-galaxy column density analysis...") if rank == 0 else None
        run_halo_column_density(cfg)

    elif getattr(cfg.omega_ion, "calculate", False):
        print("\nRunning omega parameter analysis...") if rank == 0 else None
        run_omega_parameter(cfg)
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

    if monitor_thread is not None:
        # Daemon thread will exit with the process; this join keeps the thread clean on shutdown.
        monitor_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()