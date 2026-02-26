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

#own modules
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

#the SOAP-HBT halo properties file

soap_path = str(
    Path(cfg.simulation.main_dir)
    /f"L{cfg.simulation.box_length:03d}_m{cfg.simulation.resolution}"
    /cfg.simulation.name
    /"SOAP-HBT"
    /f"halo_properties_{cfg.simulation.snapshot_number:04d}.hdf5"
)

#redshift list and output list of the simulation, to be used for loading the correct chimes table
redshift_list_path= str(Path(cfg.simulation.main_dir)
    /f"L{cfg.simulation.box_length:03d}_m{cfg.simulation.resolution}"
    /cfg.simulation.name
    /"output_list.txt")

#redshift extraction from the output list, to be used for loading the correct chimes table
redshift_list = pandas.read_csv(
    redshift_list_path,
    comment="#",
    header=None,
    names=["redshift", "type"]
)
redshift_list["type"] = redshift_list["type"].str.strip()
redshift=float(redshift_list.iloc[cfg.simulation.snapshot_number]["redshift"])
snapshot_type=redshift_list.iloc[cfg.simulation.snapshot_number]["type"]

#load the chimes abundances table for the same snapshot
chimes_path = str(
    Path(cfg.simulation.chimes_table_dir)
    /f"z{redshift:.3f}_eqm.hdf5"
)







mask = swift.mask(snapshot_path)
# The full metadata object is available from within the mask
b = mask.metadata.boxsize #boxsize

#histogram of column density of the gas particles in aforementioned window with dz mentioned above. Now for loading still comoving
resolution = 1000
xmin = 0*b[0]
xmax = 1*b[0] 
dx=(xmax-xmin)/resolution

ymin=0*b[1]
ymax=1*b[1]
dy=(ymax-ymin)/resolution

zmin = 0.0* b[2]
zmax = 0.2* b[2]
dz = (zmax - zmin)/resolution

load_region = [
        [xmin, xmax],
        [ymin, ymax],
        [zmin, zmax]
    ]

# Constrain the mask
mask.constrain_spatial(load_region)



# Now load the snapshot with this mask
snapshot = load(snapshot_path, mask=mask)


#load the gas particles
gas_particles = snapshot.gas

#hydrogen number density in cm^-3 for each gas particle
nh = chem.elements.get_particle_density(element="hydrogen",
                                        gas_particles=gas_particles,
                                        physical=True)
nh_cm3= nh.to("1/cm**3")

#temperature for each gas particle
temperatures = snapshot.gas.temperatures.to_physical()
print(temperatures[:5])#metallicity for each gas particle in units of solar metallicity
metallicities= snapshot.gas.metal_mass_fractions.to_physical()
#hybrid CHIMES paper solar metallicity (https://richings.bitbucket.io/chimes/user_guide/ChimesData/equilibrium_abundances.html)
solar_metallicity = 0.0129
metallicities = metallicities.value / solar_metallicity #in units of solar metallicity
print("minimum solar metallicity:", np.min(metallicities))

positions = snapshot.gas.coordinates.to_physical()





#column density histogram in x,y plane
counts_histogram=np.histogram2d(x=positions[:,0].to("Mpc").value, #ensure same units for positions
                                y=positions[:,1].to("Mpc").value, 
                                bins=(resolution.to("Mpc").value)**2, #squared as to ensure that the total number of bins is resolution^2, as we are doing a 2D histogram 
                                range=[[xmin, xmax], [ymin, ymax]],
                                weights=nh_cm3.value)


# Unpack histogram
particles_hist, xedges, yedges = counts_histogram

particle_column_density = particles_hist/((xedges[1]-xedges[0])*(yedges[1]-yedges[0])) #convert to column density by dividing by the area of the bin and multiplying by the depth of the box
particle_column_density = particle_column_density *.to("1/cm**2").value #convert to column density in cm^-2 by multiplying by the depth of the box in cm
#setting up the plotter with the constant edges for temperature and density, so that all the plots have the same axes and can be easily compared
plotter = plot.column_density_plotter(x_edges=xedges, y_edges=yedges)

plotter.plot_xy(column_density_values=particle_column_density, column_density_unit=r"$n_H [cm^{-2}]$", output_path="test_colibre/column_density_nH.png")
print("Finished test_colibre/column_density_nH.png")

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

plotter.plot(density_values=average_metallicity, density_unit=r"$<Z/Z_\odot>$", output_path="test_colibre/metallicity_hist.png")
print("Finished test_colibre/metallicity_hist.png")



###----- HI HISTOGRAM -----###


#now the chimes ion equilibrium densities
chimes=chem.chimes(chimes_path)
HI_abundance=chimes.extract_ion_abundance(ion="HI", 
                                        log_Z=np.log10(metallicities,
                                                        where=metallicities>0,
                                                        #make sure to avoid logZ=-inf
                                                        out=np.full_like(metallicities, np.nan, dtype=float)), 
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
    where=particles != 0
)

plotter.plot(density_values=average_HI, density_unit=r"$<n_{HI}/n_H>$", output_path="test_colibre/HI_abundance_hist.png")
print("Finished test_colibre/HI_abundance_hist.png")


