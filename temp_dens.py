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
import pandas

from spec_analysis import data_structure as ds
from spec_analysis import cosmology as cosmo
from spec_analysis import chemistry as chem
from spec_analysis import plot
from spec_analysis import unpack_data


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


#unpacking of file names
data_unpacker = unpack_data.unwrapper(cfg)
comoving_box_size = data_unpacker.box_size.to("Mpc")

zmin = 0.0* comoving_box_size
zmax = 0.2 * comoving_box_size
load_region = [
        [0.0 * comoving_box_size, 1.0 * comoving_box_size],
        [0.0 * comoving_box_size, 1.0 * comoving_box_size],
        [zmin, zmax]
    ]

snapshot = data_unpacker.load_snapshot(load_region=load_region)

gas_particles = snapshot.gas


nh_cm3 = chem.elements.get_particle_density(element="hydrogen",
                                        gas_particles=gas_particles,
                                        physical=True).to("1/cm**3")

temperatures = gas_particles.temperatures.to_physical()
metallicities= gas_particles.metal_mass_fractions.to_physical()
#hybrid CHIMES paper solar metallicity (https://richings.bitbucket.io/chimes/user_guide/ChimesData/equilibrium_abundances.html)
solar_metallicity = 0.0129
metallicities = metallicities.value / solar_metallicity #in units of solar metallicity



#histogram of temperature vs density
log_temp_min = 0
log_temp_max = 10

log_nh_cm3_min = -9
log_nh_cm3_max = 6


counts_histogram=np.histogram2d(x=np.log10(nh_cm3.value), 
                                y=np.log10(temperatures.value), 
                                bins=1000, 
                                range=[[log_nh_cm3_min, log_nh_cm3_max], [log_temp_min, log_temp_max]],
                                weights=None)


# Unpack histogram
particles_hist, xedges, yedges = counts_histogram

#setting up the plotter with the constant edges for temperature and density, so that all the plots have the same axes and can be easily compared
plotter = plot.temperature_density_plotter(density_edges=xedges, temperature_edges=yedges)

plotter.plot(density_values=particles_hist, 
            density_unit="Number of particles", 
            log_scale=True,
            title="Hydrogen temperature-density occupation",
            output_path="test_colibre/particles_hist.png")
print("Finished test_colibre/particles_hist.png")

###----- METALICITY HISTOGRAM -----###

metallicity_histogram=np.histogram2d(x=np.log10(nh_cm3.value), 
                                y=np.log10(temperatures.value), 
                                bins=1000, 
                                range=[[log_nh_cm3_min, log_nh_cm3_max], [log_temp_min, log_temp_max]],
                                weights=metallicities)

metallicities_hist, _,_ = metallicity_histogram

#Whenever there are no particles, we set the metallicity to NaN to avoid plotting issues with log scale
average_metallicity = np.divide(
    metallicities_hist,
    particles_hist,
    out=np.full_like(metallicities_hist, np.nan, dtype=float),
    where=particles_hist != 0
)

plotter.plot(density_values=average_metallicity, 
            density_unit=r"$<Z/Z_\odot>$", 
            log_scale=False,
            title="Metallicity",
            output_path="test_colibre/metallicity_hist.png")
print("Finished test_colibre/metallicity_hist.png")



###----- HI HISTOGRAM -----###


#now the chimes ion equilibrium densities
chimes=chem.chimes(data_unpacker.chimes_table_path)
HI_abundance=chimes.extract_ion_abundance(ion="HI", 
                                        log_Z=np.log10(metallicities,
                                                        where=metallicities>0,
                                                        #make sure to avoid logZ=-inf, so set this at -40 which is much lower than the lowest logZ in the table
                                                        out=np.full_like(metallicities, -40, dtype=float)), 
                                        log_T=np.log10(temperatures), 
                                        log_nH_cm3=np.log10(nh_cm3)
                                        )
HI_histogram=np.histogram2d(x=np.log10(nh_cm3.value), 
                            y=np.log10(temperatures.value), 
                            bins=1000, 
                            range=[[log_nh_cm3_min, log_nh_cm3_max], [log_temp_min, log_temp_max]],
                            weights=HI_abundance)
# Unpack histogram
HI_abundance_hist, _,_ = HI_histogram

average_HI = np.divide(
    HI_abundance_hist,
    particles_hist,
    out=np.full_like(HI_abundance_hist, fill_value=np.nan, dtype=float),
    where=particles_hist != 0
)

plotter.plot(density_values=average_HI,
             density_unit=r"$<Log_{10}(n_{HI}/n_H)>$",
             log_scale=False, 
             title="HI abundance",
             output_path="test_colibre/HI_abundance_hist.png")
print("Finished test_colibre/HI_abundance_hist.png")