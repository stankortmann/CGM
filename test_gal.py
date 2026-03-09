from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as swift
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity
import h5py 
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import argparse
import yaml
import unyt as u
import pandas

#own modules
from spec_analysis import data_structure as ds
from spec_analysis import cosmology as cosmo
from spec_analysis import chemistry as chem
from spec_analysis import plot
from spec_analysis import unpack_data
from spec_analysis import density_profiles
from spec_analysis import galaxy_selection as gal_sel

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
        monitoring=ds.Monitoring(**cfg_dict['monitoring']),
        window=ds.Window(**cfg_dict['window']),
        chemistry=ds.Chemistry(**cfg_dict['chemistry']),
        galaxy=ds.Galaxy(**cfg_dict['galaxy'])
    )




#unpacking
data_unpacker = unpack_data.unwrapper(cfg)


single_galaxy=gal_sel.single_galaxy(cfg=cfg, data_unpacker=data_unpacker)
print("single galaxy selected")
#we now add an extra field to cfg.galaxy to get proper galaxy extend window
cfg.galaxy.extend= cosmo_quantity(cfg.galaxy.extend_value,
                                u.Unit(cfg.galaxy.extend_unit),
                                comoving=False,
                                scale_factor=data_unpacker.scale_factor,            
                                cosmo_factor=1
                                )

# Project density (mass) along one axis (say z→ surface density on x-y)
surface_density = project_gas(
    gas_in_halo_properties,
    resolution=100,
    
    project="masses",
    periodic=True,
    parallel=True
)
print("projection done")
###### without haloes ######
plt.figure(figsize=(8,8))
plt.imshow(
    surface_density.to_physical_value("Msun/Mpc**2").T,  # convert if units are comoving
    origin="lower",
    norm=LogNorm(),
    cmap="inferno"
)
plt.colorbar(label=f"Surface density [{str(surface_density.units)}]")

plt.xlabel("x [Mpc comoving]")
plt.ylabel("y [Mpc comoving]")
plt.title("Projected surface density (along z)")
plt.tight_layout()
plt.savefig("zz_galaxy.png", dpi=300)
print("done galaxy masses test")
plt.close()


