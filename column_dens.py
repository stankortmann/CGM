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
from spec_analysis import column_density

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
        chemistry=ds.Chemistry(**cfg_dict['chemistry'])
    )





data_unpacker = unpack_data.unwrapper(cfg)
comoving_box_size = data_unpacker.box_size.to("Mpc")
cfg.window.x,cfg.window.y,cfg.window.z = [x*comoving_box_size for x in cfg.window.x], [y*comoving_box_size for y in cfg.window.y], [z*comoving_box_size for z in cfg.window.z]

dx=(cfg.window.x[1]-cfg.window.x[0])/cfg.window.resolution
dy=(cfg.window.y[1]-cfg.window.y[0])/cfg.window.resolution
dz=(cfg.window.z[1]-cfg.window.z[0])/cfg.window.resolution

region = [
        cfg.window.x,
        cfg.window.y,
        cfg.window.z
    ]

snapshot = data_unpacker.load_snapshot(load_region=region)

gas_particles = snapshot.gas


#number of physical hydrogen particles of each gas particle
n_element =chem.elements.get_particle_number(cfg.chemistry.element, gas_particles).value

#element density for each gas particle
n_element_cm3 = chem.elements.get_particle_density(element=cfg.chemistry.element,
                                        gas_particles=gas_particles,
                                        physical=True).to("1/cm**3")

#We always need the hydrogen number density for the CHIMES table, even if we are not looking at hydrogen
if cfg.chemistry.ion != "hydrogen":
    n_H_cm3 = chem.elements.get_particle_density(element="hydrogen",
                                        gas_particles=gas_particles,
                                        physical=True).to("1/cm**3")
else:
    n_H_cm3 = n_element_cm3

#temperature for each gas particle
temperatures = gas_particles.temperatures.to_physical()
#metallicity for each gas particle
metallicities= gas_particles.metal_mass_fractions.to_physical()
#hybrid CHIMES paper solar metallicity (https://richings.bitbucket.io/chimes/user_guide/ChimesData/equilibrium_abundances.html)
solar_metallicity = 0.0129
metallicities = metallicities.value / solar_metallicity #in units of solar metallicity

positions = gas_particles.coordinates.to_physical()




#column density histogram in x,y plane
n_element_hist=np.histogram2d(x=positions[:,0].to("Mpc").value, #ensure same units for positions
                                y=positions[:,1].to("Mpc").value, 
                                bins=(cfg.window.resolution,cfg.window.resolution), #squared as to ensure that the total number of bins is resolution^2, as we are doing a 2D histogram 
                                range=[[float(cfg.window.x[0].to("Mpc").to_physical().value), float(cfg.window.x[1].to("Mpc").to_physical().value)],
                                     [float(cfg.window.y[0].to("Mpc").to_physical().value), float(cfg.window.y[1].to("Mpc").to_physical().value)]],
                                weights=n_element)


# Unpack histogram
n_element_counts, xedges, yedges = n_element_hist

n_element_column_density = n_element_counts/(dx.to_physical()*dy.to_physical()) #convert to column density by dividing by the area of the bin and multiplying by the depth of the box
n_element_column_density = n_element_column_density.to("1/cm**2").value #convert to column density in cm^-2 by multiplying by the depth of the box in cm
#setting up the plotter with the constant edges for temperature and density, so that all the plots have the same axes and can be easily compared
plotter = plot.column_density_plotter(x_edges=xedges, y_edges=yedges)

plotter.plot_xy(column_density_values=n_element_column_density, 
                column_density_unit=r"$n_{%s} [cm^{-2}]$" % cfg.chemistry.element,
                title="Column density of %s in x-y plane" % cfg.chemistry.element,
                log_scale=True, 
                output_path="test_colibre/column_density_%s.png" % cfg.chemistry.element)
print("Finished test_colibre/column_density_%s.png" % cfg.chemistry.element)

###----- HI HISTOGRAM -----###


#now the chimes ion equilibrium densities
chimes=chem.chimes(data_unpacker.chimes_table_path)
log_ion_element_fraction=chimes.extract_ion_abundance(ion=cfg.chemistry.ion, 
                                        log_Z=np.log10(metallicities,
                                                        where=metallicities>0,
                                                        #make sure to avoid logZ=-inf, so set this at -40 which is much lower than the lowest logZ in the table
                                                        out=np.full_like(metallicities, -40, dtype=float)), 
                                        log_T=np.log10(temperatures), 
                                        log_n_H_cm3=np.log10(n_H_cm3)
                                        )
n_ion = n_element * 10**log_ion_element_fraction #number of HI particles for each gas particle

n_ion_hist=np.histogram2d(x=positions[:,0].to("Mpc").value, #ensure same units for positions
                                y=positions[:,1].to("Mpc").value, 
                                bins=(cfg.window.resolution,cfg.window.resolution), #squared as to ensure that the total number of bins is resolution^2, as we are doing a 2D histogram 
                                range=[[float(cfg.window.x[0].to("Mpc").to_physical().value), float(cfg.window.x[1].to("Mpc").to_physical().value)],
                                     [float(cfg.window.y[0].to("Mpc").to_physical().value), float(cfg.window.y[1].to("Mpc").to_physical().value)]],
                                weights=n_ion)
# Unpack histogram
n_ion_counts, _, _ = n_ion_hist

n_ion_column_density = n_ion_counts/(dx.to_physical()*dy.to_physical()) #convert to column density by dividing by the area of the bin and multiplying by the depth of the box
n_ion_column_density = n_ion_column_density.to("1/cm**2").value #convert to column density in cm^-2 by multiplying by the depth of the box in cm
#setting up the plotter with the constant edges for temperature and density, so that all the plots have the same axes and can be easily compared

plotter.plot_xy(column_density_values=n_ion_column_density, 
                column_density_unit=r"$n_{%s} [cm^{-2}]$" % cfg.chemistry.ion,
                title="Column density of %s in x-y plane" % cfg.chemistry.ion,
                log_scale=True, 
                output_path="test_colibre/column_density_%s.png" % cfg.chemistry.ion)
print("Finished test_colibre/column_density_%s.png" % cfg.chemistry.ion)



