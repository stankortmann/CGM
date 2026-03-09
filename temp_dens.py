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
from fast_histogram import histogram2d

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
        default="configurations/test_box.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # --- Load YAML config file ---
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Create config object
    cfg = ds.Config(
        simulation=ds.Simulation(**cfg_dict["simulation"]),
        data_output=ds.Data_output(**cfg_dict["data_output"]),
        monitoring=ds.Monitoring(**cfg_dict["monitoring"]),
        window=ds.Window(**cfg_dict["window"]),
        chemistry=ds.Chemistry(**cfg_dict["chemistry"]),
        galaxy=ds.Galaxy(**cfg_dict["galaxy"]),
    )

    # --- Unpack data ---
    data_unpacker = unpack_data.unwrapper(cfg)
    comoving_box_size = data_unpacker.box_size.to("Mpc")

    zmin = 0.0 * comoving_box_size
    zmax = 0.05 * comoving_box_size

    load_region = [
        [0.0 * comoving_box_size, 1.0 * comoving_box_size],
        [0.0 * comoving_box_size, 1.0 * comoving_box_size],
        [zmin, zmax],
    ]

    snapshot = data_unpacker.load_snapshot(load_region=load_region)
    gas_particles = snapshot.gas
    print("gas particles are loaded")
    
    

    temperatures = gas_particles.temperatures.to_physical()
    print(np.shape(temperatures))
    metallicities = gas_particles.metal_mass_fractions.to_physical()
    nh_cm3 = chem.elements.get_particle_density(
        element="hydrogen",
        gas_particles=gas_particles,
        physical=True,
    ).to("1/cm**3")
    print("number density of hydrogen has been calculated")
    # hybrid CHIMES paper solar metallicity
    solar_metallicity = 0.0129
    metallicities = metallicities.value / solar_metallicity

    # --- Histogram of temperature vs density ---
    nbins=1000
    log_temp_min = 0
    log_temp_max = 10

    log_nh_cm3_min = -9
    log_nh_cm3_max = 6

    density_edges = np.linspace(log_nh_cm3_min, log_nh_cm3_max, nbins + 1)
    temperature_edges = np.linspace(log_temp_min, log_temp_max, nbins + 1)

    log_nh=np.log10(nh_cm3.value)
    log_T=np.log10(temperatures.value)
    log_Z=np.log10(
                metallicities,
                where=metallicities > 0,
                out=np.full_like(metallicities, -40, dtype=float))

    particles_hist = histogram2d(
        log_nh,
        log_T,
        bins=[nbins, nbins],
        range=[
                [log_nh_cm3_min, log_nh_cm3_max],
                [log_temp_min, log_temp_max],
                ]
                )

    

    # Plotter setup
    plotter = plot.temperature_density_plotter(
        density_edges=density_edges, temperature_edges=temperature_edges
    )

    plotter.plot(
        density_values=particles_hist,
        density_unit="Number of particles",
        log_scale=True,
        title="Hydrogen temperature-density occupation",
        output_path="test_colibre/particles_hist.png",
    )

    print("Finished test_colibre/particles_hist.png")

    # --- Metallicity histogram ---
    metallicities_hist= histogram2d(
        x=log_nh,
        y=log_T,
        bins=[nbins,nbins],
        range=[[log_nh_cm3_min, log_nh_cm3_max],
                [log_temp_min, log_temp_max]],
        weights=metallicities
    )


    average_metallicity = np.divide(
        metallicities_hist,
        particles_hist,
        out=np.full_like(metallicities_hist, np.nan, dtype=float),
        where=particles_hist != 0,
    )

    plotter.plot(
        density_values=average_metallicity,
        density_unit=r"$<Z/Z_\odot>$",
        log_scale=False,
        title="Metallicity",
        output_path="test_colibre/metallicity_hist.png",
    )

    print("Finished test_colibre/metallicity_hist.png")

    # --- Ion histogram ---
    ions = ["HI", "Hm", "HII"]

    chimes = chem.chimes(data_unpacker.chimes_table_path)

    for ion in ions:

        ion_abundance = chimes.extract_ion_abundance(
            ion=ion,
            log_Z=log_Z,
            log_T=log_T,
            log_n_H_cm3=log_nh,
        )

        ion_abundance_hist = histogram2d(
            x=log_nh,
            y=log_T,
            bins=[nbins,nbins],
            range=[[log_nh_cm3_min, log_nh_cm3_max],
                 [log_temp_min, log_temp_max]],
            weights=ion_abundance
        )

        

        average_ion = np.divide(
            ion_abundance_hist,
            particles_hist,
            out=np.full_like(ion_abundance_hist, fill_value=np.nan, dtype=float),
            where=particles_hist != 0,
        )

        plotter.plot(
            density_values=average_ion,
            density_unit=rf"$<Log_{{10}}(n_{{{ion}}}/n_H)>$",
            log_scale=False,
            title=f"{ion} abundance",
            output_path=f"test_colibre/{ion}_abundance_hist.png",
        )

        print(f"Finished test_colibre/{ion}_abundance_hist.png")

#exit the python environment
exit()