import numpy as np
import unyt as u
from functools import cached_property

#own modules
from spec_analysis import chemistry as chem
from spec_analysis import plot

from functools import cached_property
import numpy as np
import unyt as u


class column_density_2d:
    """
    Computes 2D column density histograms for elements and their ions.
    Designed for SWIFT + CHIMES workflows.
    """

    def __init__(self, cfg, length_unit, filenames, gas_particles, element, halo=None):
        self.cfg = cfg
        self.gas_particles = gas_particles
        self.element = element
        self.halo = halo

        self.chimes = chem.chimes(filenames.chimes_table_path)

        self.length_unit=length_unit



    @cached_property
    def n_H_cm3(self):
        return chem.elements.get_particle_density(
            element="hydrogen",
            gas_particles=self.gas_particles,
            physical=True
        ).to("1/cm**3")

    @cached_property
    def temperatures(self):
        return self.gas_particles.temperatures.to_physical()

    @cached_property
    def metallicities(self):
        solar_metallicity = 0.0129
        Z = self.gas_particles.metal_mass_fractions.to_physical().value
        return Z / solar_metallicity

    @cached_property
    def positions(self):
        return self.gas_particles.coordinates.to(self.length_unit).to_physical()


    @cached_property
    def histogram_range(self):
        barrier = self.cfg.galaxy.extend.to(self.length_unit)

        if self.cfg.galaxy.single_galaxy:
            x0 = self.halo.position[:, 0].to(self.length_unit).to_physical().value
            y0 = self.halo.position[:, 1].to(self.length_unit).to_physical().value
            return [[x0 - barrier, x0 + barrier],
                    [y0 - barrier, y0 + barrier]]

        return [[float(self.cfg.window.x[0].to(self.length_unit).to_physical().value),
                 float(self.cfg.window.x[1].to(self.length_unit).to_physical().value)],
                [float(self.cfg.window.y[0].to(self.length_unit).to_physical().value),
                 float(self.cfg.window.y[1].to(self.length_unit).to_physical().value)]]



    @cached_property
    def histogram_data(self):
        x = self.positions[:, 0].value
        y = self.positions[:, 1].value

        hist, xedges, yedges = np.histogram2d(
            x=x,
            y=y,
            bins=(self.cfg.window.resolution,
                self.cfg.window.resolution),
            range=self.histogram_range,
        )

        return hist, xedges, yedges
    
    @cached_property
    def xedges(self):
        return self.histogram_data[1]


    @cached_property
    def yedges(self):
        return self.histogram_data[2]

    
    @cached_property
    def pixel_area(self):
        dx = self.xedges[1] - self.xedges[0]
        dy = self.yedges[1] - self.yedges[0]

        unit_x = self.positions.unit[0]
        unit_y = self.positions.unit[1]

        return dx * dy * unit_x * unit_y

    @cached_property
    def n_element(self):
        return chem.elements.get_particle_number(
            self.element,
            self.gas_particles
        ).value

    @cached_property
    def n_element_cm3(self):
        return chem.elements.get_particle_density(
            element=self.element,
            gas_particles=self.gas_particles,
            physical=True
        ).to("1/cm**3")

    @cached_property
    def element_column_density(self):
        x = self.positions[:, 0].value
        y = self.positions[:, 1].value

        hist, _, _ = np.histogram2d(
            x=x,
            y=y,
            bins=(self.cfg.window.resolution,
                  self.cfg.window.resolution),
            range=self.histogram_range,
            weights=self.n_element
        )

        cd = hist / self.pixel_area
        return cd.to("1/cm**2")


    def column_density_ion(self, ion):

        log_frac = self.chimes.extract_ion_abundance(
            ion=ion,
            log_Z=np.log10(
                self.metallicities,
                where=self.metallicities > 0,
                out=np.full_like(self.metallicities, -40.0)
            ),
            log_T=np.log10(self.temperatures),
            log_n_H_cm3=np.log10(self.n_H_cm3),
        )

        n_ion = self.n_element * 10**log_frac

        x = self.positions[:, 0].value
        y = self.positions[:, 1].value

        hist, _, _ = np.histogram2d(
            x=x,
            y=y,
            bins=(self.cfg.window.resolution,
                  self.cfg.window.resolution),
            range=self.histogram_range,
            weights=n_ion
        )

        cd = hist / self.pixel_area
        return cd.to("1/cm**2")

    
    def column_density_distribution_function(
        self,
        ion=None,
        log_column_density_range=None,
        n_bins=100,
        normalize=True
    ):

        if ion is None:
            cd = self.element_column_density
        else:
            cd = self.column_density_ion(ion)
        
        
        values = cd.to("1/cm**2").value.flatten()
        values = values[values > 0]

        log_values = np.log10(values)

        if log_column_density_range is None:
            log_column_density_range = [
                log_values.min(),
                log_values.max()
            ]

        hist, edges = np.histogram(
            log_values,
            bins=n_bins,
            range=log_column_density_range,
            density=normalize
        )

        dlog = edges[1] - edges[0]
        centers = 0.5 * (edges[1:] + edges[:-1])

        return hist, centers, dlog


class column_density_1d:
    """
    Computes radial (transverse) column density profiles
    for elements and ions relative to a halo centre.

    Designed for SWIFT + CHIMES workflows.
    """

    def __init__(self, cfg, length_unit, filenames, gas_particles, element, halo=None):
        self.cfg = cfg
        self.gas_particles = gas_particles
        self.element = element
        self.halo = halo
        self.length_unit=length_unit
        self.chimes = chem.chimes(filenames.chimes_table_path)

    # ==========================================================
    # Basic particle properties (cached like 2D case)
    # ==========================================================

    @cached_property
    def n_H_cm3(self):
        return chem.elements.get_particle_density(
            element="hydrogen",
            gas_particles=self.gas_particles,
            physical=True
        ).to("1/cm**3")

    @cached_property
    def temperatures(self):
        return self.gas_particles.temperatures.to_physical()

    @cached_property
    def metallicities(self):
        solar_metallicity = 0.0129
        Z = self.gas_particles.metal_mass_fractions.to_physical().value
        return Z / solar_metallicity

    @cached_property
    def positions(self):
        return self.gas_particles.coordinates.to(self.length_unit).to_physical()

    # ==========================================================
    # Geometry
    # ==========================================================

    @cached_property
    def halo_center(self):
        if self.halo is None:
            raise ValueError("Halo must be provided for radial profiles.")

        x0 = self.halo.position[:, 0].to(self.length_unit).to_physical().value
        y0 = self.halo.position[:, 1].to(self.length_unit).to_physical().value
        return x0, y0

    @cached_property
    def transverse_distance(self):
        """
        Projected (x-y plane) distance from halo centre.
        Returns unyt array in Mpc.
        """
        x = self.positions[:, 0].value
        y = self.positions[:, 1].value

        x0, y0 = self.halo_center

        dx = x - x0
        dy = y - y0

        r = np.sqrt(dx**2 + dy**2)

        return r * u.Mpc

    # ==========================================================
    # Element properties
    # ==========================================================

    @cached_property
    def n_element(self):
        return chem.elements.get_particle_number(
            self.element,
            self.gas_particles
        ).value

    @cached_property
    def n_element_cm3(self):
        return chem.elements.get_particle_density(
            element=self.element,
            gas_particles=self.gas_particles,
            physical=True
        ).to("1/cm**3")

    # ==========================================================
    # Ion number per particle
    # ==========================================================

    def n_ion(self, ion):

        log_frac = self.chimes.extract_ion_abundance(
            ion=ion,
            log_Z=np.log10(
                self.metallicities,
                where=self.metallicities > 0,
                out=np.full_like(self.metallicities, -40.0)
            ),
            log_T=np.log10(self.temperatures),
            log_n_H_cm3=np.log10(self.n_H_cm3),
        )

        return self.n_element * 10**log_frac

    # ==========================================================
    # Radial Column Density Profile
    # ==========================================================

    def radial_column_density_profile(
        self,
        ion=None,
        r_max=500*u.kpc,        #Kpc proper length
        n_bins=50,
        log_bins=False
    ):
        """
        Compute radial column density profile N(r).
        """

        r = self.transverse_distance.to(self.length_unit).value
        r_max=r_max.to(self.length_unit).value
        # Select weights
        if ion is None:
            weights = self.n_element
        else:
            weights = self.n_ion(ion)

        # Define bins
        if log_bins:
            bins = np.logspace(
                np.log10(r[r > 0].min()),
                np.log10(r_max),
                n_bins + 1
            )
        else:
            bins = np.linspace(0, r_max, n_bins + 1)

        counts, edges = np.histogram(
            r,
            bins=bins,
            weights=weights
        )

        # Annulus area
        r_outer = edges[1:]
        r_inner = edges[:-1]

        area = np.pi * (r_outer**2 - r_inner**2)
        area = area * (self.length_unit**2)

        column_density = counts / area
        column_density = column_density.to("1/cm**2")

        r_centers = 0.5 * (r_outer + r_inner)

        return r_centers, column_density

    # ==========================================================
    # Radial Column Density Distribution Function
    # ==========================================================

    def column_density_distribution_function(
        self,
        ion=None,
        r_max=0.5,
        n_radial_bins=50,
        n_cddf_bins=100,
        normalize=True
    ):

        r_centers, cd_profile = self.radial_column_density_profile(
            ion=ion,
            r_max=r_max,
            n_bins=n_radial_bins
        )

        values = cd_profile.to("1/cm**2").value
        values = values[values > 0]

        log_values = np.log10(values)

        hist, edges = np.histogram(
            log_values,
            bins=n_cddf_bins,
            density=normalize
        )

        dlog = edges[1] - edges[0]
        centers = 0.5 * (edges[1:] + edges[:-1])

        return hist, centers, dlog



        
        
        





