from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as swift
from swiftsimio.visualisation.projection import project_gas
import h5py 
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import argparse
import yaml
import unyt as u

from spec_analysis import data_structure as ds
from spec_analysis import cosmology as cosmo
from spec_analysis import chemistry as chem

if __name__ == "__main__":

      # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Run CGM analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/test.yaml",
        help="Path to the YAML configuration file"
    )
    args=parser.parse_args()
    # --- Load YAML config file ---
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Create config object
    cfg = ds.Config(
        simulation=ds.Simulation(**cfg_dict['simulation']),
        data_output=ds.Data_output(**cfg_dict['data_output']),
        monitoring=ds.Monitoring(**cfg_dict['monitoring'])    
    )







# Load snapshot

snapshot_path = str(
    Path(cfg.simulation.main_dir)
    /f"L{cfg.simulation.box_length:03d}_m{cfg.simulation.resolution}"
    /cfg.simulation.name
    /"snapshots"
    /f"colibre_{cfg.simulation.snapshot_number:04d}"
    /f"colibre_{cfg.simulation.snapshot_number:04d}.hdf5"
)


#load the SOAP-HBT catalog for the same snapshot

soap_path = str(
    Path(cfg.simulation.main_dir)
    /f"L{cfg.simulation.box_length:03d}_m{cfg.simulation.resolution}"
    /cfg.simulation.name
    /"SOAP-HBT"
    /f"halo_properties_{cfg.simulation.snapshot_number:04d}.hdf5"
)

#colibre resolution gas particle masses in Msun
gas_mass_dict = {
    5: 2.30e5 * u.Msun,
    6: 1.84e6 * u.Msun,
    7: 1.47e7 * u.Msun,
}

gas_particle_mass = gas_mass_dict[cfg.simulation.resolution]

mask = swift.mask(snapshot_path)
# The full metadata object is available from within the mask
b = mask.metadata.boxsize #boxsize
# load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
zmin = 0.2* b[2]
zmax = 0.9 * b[2]
#load_region = [[0.0 * b, 0.5 * b] for b in boxsize]
load_region = [
        [0.0 * b[0], 1.0 * b[0]],
        [0.0 * b[1], 1.0 * b[1]],
        [zmin, zmax]
    ]

# Constrain the mask
mask.constrain_spatial(load_region)



# Now load the snapshot with this mask
snapshot = load(snapshot_path, mask=mask)
densities = snapshot.gas.densities.to_physical()

nh = densities*chem.elements.particles_per_mass("hydrogen")
nh_cm3= nh.to("1/cm**3")
print(np.min(nh_cm3))
print(np.max(nh_cm3))





