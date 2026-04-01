import numpy as np
import unyt as u
from functools import cached_property
from fast_histogram import histogram2d
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.objects import cosmo_array, cosmo_factor

#own modules
from spec_analysis import chemistry as chem
from spec_analysis import plot
from spec_analysis import cosmology as cosmo




class column_density_2d:
    """
    Computes 2D column density histograms for elements and their ions.
    Designed for SWIFT + CHIMES workflows.
    """

    def __init__(self, cfg, length_unit, data_unpacker, gas_particles, element, halo=None):
        self.cfg = cfg
        self.gas_particles = gas_particles
        self.element = element
        self.halo = halo

        self.chimes = chem.chimes(data_unpacker.chimes_table_path)

        self.length_unit=length_unit



    @cached_property
    def n_H_cm3(self):
        return chem.elements.get_particle_density(
            element="hydrogen",
            gas_particles=self.gas_particles,
            physical=True
        ).to("1/cm**3")

    @cached_property
    def log_n_H_cm3(self):
        return np.log10(self.n_H_cm3.value)

    @cached_property
    def temperatures(self):
        return self.gas_particles.temperatures.to_physical()

    @cached_property
    def log_T(self):
        return np.log10(self.temperatures.value)

    @cached_property
    def metallicities(self):
        solar_metallicity = 0.0129
        Z = self.gas_particles.metal_mass_fractions.to_physical().value
        if self.cfg.chemistry.metallicity:
            return Z / solar_metallicity
        else:
            return np.full_like(Z, 0.1)  # Use solar metallicity if not taking it into account

    @cached_property
    def log_Z(self):
        log_Z=np.log10(
                self.metallicities,
                where=self.metallicities > 0,
                out=np.full_like(self.metallicities, -40.0)
                )
        return log_Z

    @cached_property
    def positions(self):
        return self.gas_particles.coordinates.to(self.length_unit).to_physical()


    @cached_property
    def histogram_range(self):
        

        if self.cfg.galaxy.single_galaxy:
            barrier = self.cfg.galaxy.extend.to(self.length_unit)
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
        xrange=self.histogram_range[0]
        yrange=self.histogram_range[1]

        return xrange, yrange
    
    @cached_property
    def xedges(self):
        xmin=self.histogram_data[0][0]
        xmax=self.histogram_data[0][1]
        xedges=np.linspace(xmin,xmax,self.cfg.window.resolution+1)
        return xedges


    @cached_property
    def yedges(self):
        ymin=self.histogram_data[1][0]
        ymax=self.histogram_data[1][1]
        yedges=np.linspace(ymin,ymax,self.cfg.window.resolution+1)
        return yedges

    
    @cached_property
    def pixel_area(self):
        dx = self.xedges[1] - self.xedges[0]
        dy = self.yedges[1] - self.yedges[0]
        return dx * dy * u.Unit(self.length_unit)**2

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
       

        hist= histogram2d(
            x=x,
            y=y,
            bins=[self.cfg.window.resolution,
                  self.cfg.window.resolution],
            range=self.histogram_range,
            weights=self.n_element
        )

        cd = hist / self.pixel_area
        return cd.to("1/cm**2").to_physical().value


    def column_density_ion(self, ion):

        log_frac = self.chimes.extract_ion_abundance(
            ion=ion,
            log_Z=self.log_Z,
            log_T=self.log_T,
            log_n_H_cm3=self.log_n_H_cm3,
        )

        n_ion = self.n_element * 10**log_frac

        x = self.positions[:, 0].value
        y = self.positions[:, 1].value

        hist=histogram2d(
            x=x,
            y=y,
            bins=[self.cfg.window.resolution,
                  self.cfg.window.resolution],
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
        los_range=[0,1], #projection axis length, still comoving
        ):

        if ion is None:
            cd = self.element_column_density
        else:
            cd = self.column_density_ion(ion)
        
        
        values = cd.to("1/cm**2").value.flatten()
        values = values[values > 0]
        #valid sightlines

        log_values = np.log10(values)

        #for now it is simple but this has to become a function to 
        #calculate line of sight range
        los_distance=(los_range[1]-los_range[0]).to("cm").to_physical().value

        if log_column_density_range is None:
            log_column_density_range = [
                log_values.min(),
                log_values.max()
            ]

        hist, edges = np.histogram(
            log_values,
            bins=n_bins,
            range=log_column_density_range
        )

        dlog = edges[1] - edges[0]
        centers = 0.5 * (edges[1:] + edges[:-1])
        #f(N)=d^2(N)/dlog(N)*dX, dX=los_distance
        cddf=hist/(dlog*los_distance)

        return cddf, centers, dlog

class column_density_2d_swift:
    """
    Computes 2D column density histograms for elements and their ions.
    Designed for SWIFT + CHIMES workflows.
    """

    def __init__(self, cfg, data_unpacker,chimes, 
            element, snapshot=None,gas_particles=None, halo=None,mpi=False):
        
        
        self.cfg = cfg
        self.cosmo=cosmo.cosmo_tools(data_unpacker=data_unpacker,cfg=cfg)
        
        #full snapshot
        if snapshot is not None and halo is None:
            self.snapshot=snapshot
            self.periodic=True

       
        #single galaxy 
        elif snapshot is None and halo is not None :
            self.snapshot=halo.snapshot
            self.halo=halo #contains all the info of the galaxy like half mass radius etc.
            self.periodic=False

        

        elif snapshot is None and halo is None:
            print("ERROR: No gas particles received.")
            exit()
        elif snapshot is not None and halo is not None:
            print("ERROR: Single halo and full box particles received.")
            exit()
        if mpi:
            self.threader=False #we will use MPI parallelisation, not threading
        else:
            self.threader=True #use threading for projection parallelisation, not MPI
            
        #connect gas_particles to the full snapshot to enable project_gas function
        self.gas_particles = self.snapshot.gas
        self.element = element
        

        self.chimes = chimes

        



    @cached_property
    def n_H_cm3(self):
        return chem.elements.get_particle_density(
            element="hydrogen",
            gas_particles=self.gas_particles,
            physical=True
        ).to("1/cm**3")

    @cached_property
    def log_n_H_cm3(self):
        return np.log10(self.n_H_cm3.value)

    @cached_property
    def temperatures(self):
        return self.gas_particles.temperatures.to_physical()

    @cached_property
    def log_T(self):
        return np.log10(self.temperatures.value)

    @cached_property
    def metallicities(self):
        solar_metallicity = 0.0129
        
        if self.cfg.chemistry.metallicity:
            Z = self.gas_particles.metal_mass_fractions.to_physical().value
            return Z / solar_metallicity
        else:
            return 0.1  # Use solar metallicity if not taking it into account

    @cached_property
    def log_Z(self):
        log_Z=np.log10(
                self.metallicities,
                where=self.metallicities > 0,
                out=np.full_like(self.metallicities, -40.0)
                )
        return log_Z


    @cached_property
    def projection_range(self):
        
        #not yet redefined, we have to make a cosma_array of the 
        #self.cfg.galaxy.single_galaxy unyt array
        if self.cfg.galaxy.single_galaxy:
            #stored as physical property, please get to comoving
            barrier = self.cfg.galaxy.extend.to_comoving()
            
            return [ - barrier, 
                     + barrier,
                    - barrier,
                      + barrier]
        else:
            return [self.cfg.window.x[0],
                    self.cfg.window.x[1],
                    self.cfg.window.y[0],
                    self.cfg.window.y[1]]

    @cached_property
    def xedges(self):
        xmin=self.projection_range[0]
        xmax=self.projection_range[1]
        unit_length=xmin.units
        xedges=np.linspace(xmin.value,xmax.value,self.cfg.window.resolution+1)
        xedges_cosmo_array=cosmo_array(xedges,
                                        unit_length,
                                        comoving=True,
                                        scale_factor=self.snapshot.metadata.scale_factor, 
                                        scale_exponent=1,  # distances scale as a**1, so the scale exponent is 1
                                        )
        return xedges_cosmo_array


    @cached_property
    def yedges(self):
        ymin=self.projection_range[2]
        ymax=self.projection_range[3]
        unit_length=ymin.units
        yedges=np.linspace(ymin.value,ymax.value,self.cfg.window.resolution+1)
        yedges_cosmo_array=cosmo_array(yedges,
                                        unit_length,
                                        comoving=True,
                                        scale_factor=self.snapshot.metadata.scale_factor, 
                                        scale_exponent=1,  # distances scale as a**1, so the scale exponent is 1
                                        )
        return yedges_cosmo_array

    @cached_property
    def n_element(self):
        n_element=chem.elements.get_particle_number(self.element,self.gas_particles,metallicity=self.cfg.chemistry.metallicity).value
        
        n_element_cosmo_array= cosmo_array(
                                        n_element,
                                        None,
                                        comoving=False,
                                        scale_factor=self.snapshot.metadata.scale_factor,  
                                        scale_exponent=0,  
                                        )
        return n_element_cosmo_array

    

    @cached_property
    def element_column_density(self):
        #ensure comoving as the snapshot data is in comoving coordinates
        element_number=self.n_element.to_comoving() 
        #just take the median, rescaling ensures no float overflow
        scale = np.percentile(element_number.value, 90)
        if scale <= 0:
            scale = 1.0
        scaled_element_number=element_number/scale
        self.gas_particles.element_number=scaled_element_number
        scale_projection = project_gas(
            data=self.snapshot,
            project="element_number",
            resolution=self.cfg.window.resolution,
            region=self.projection_range,
            parallel=self.threader,
            periodic=self.periodic,
        )
        projection = scale_projection*scale
        return projection.to("1/cm**2")


    def column_density_ion(self, ion):
        
        # Check if the ion fraction is directly available in the snapshot
        species = self.gas_particles.species_fractions

        if hasattr(species, ion):
            # Use the ion fraction from the simulation output
            ion_fraction = getattr(species, ion)

            n_ion = self.n_element.to_comoving() * ion_fraction

        else:
            # Fall back to CHIMES ionization table
            log_frac = self.chimes.extract_ion_abundance(
                ion=ion,
                log_Z=self.log_Z,
                log_T=self.log_T,
                log_n_H_cm3=self.log_n_H_cm3,
            )

            n_ion = self.n_element.to_comoving() * 10**log_frac
            
        scale = np.percentile(n_ion.value, 90)
        if scale <= 0:
            scale = 1.0
        n_ion_scale=n_ion/scale
        name_ion = f"{ion}_number"

        setattr(
            self.gas_particles,
            name_ion,
            n_ion_scale
        )
        scale_projection = project_gas(
            data=self.snapshot,
            project=name_ion,
            resolution=self.cfg.window.resolution,
            region=self.projection_range,
            parallel=self.threader,
            periodic=self.periodic,
        )
        projection=scale_projection*scale
        return projection.to("1/cm**2")
    
    def column_density_distribution_function(
        self,
        column_density=None,
        log_column_density_range=None,
        n_bins=100,
        los_range=[0,1], #projection axis length, still comoving
        ):
        
        cd=column_density.to_physical() #just to be sure
        
        
        values = cd.value.flatten()
        
        N_pixels=len(values)
        #valid sightlines get logged
        log_values = np.log10(values[values > 0])

        #for now it is simple but this has to become a function to 
        #calculate line of sight range
        los_distance=los_range[1]-los_range[0]
        dX_single=self.cosmo.dX(column_length=los_distance)
        dX_total=N_pixels*dX_single

        if log_column_density_range is None:
            log_column_density_range = [
                log_values.min(),
                log_values.max()
            ]

        hist, edges = np.histogram(
            log_values,
            bins=n_bins,
            range=log_column_density_range
        )

        dlogN = edges[1] - edges[0]

        centers = 0.5 * (edges[1:] + edges[:-1])
        N = 10**centers

        cddf = hist / (dlogN * np.log(10) * N * dX_total)

        return cddf, centers, dlogN

    @cached_property
    def transverse_distance(self):
        """
        Determines distance to the centre of the halo/galaxy of each pixel
        """
        
        x_center_pixel = 0.5 * (self.xedges[1:] + self.xedges[:-1])
        y_center_pixel = 0.5 * (self.yedges[1:] + self.yedges[:-1])
        grid_x, grid_y = np.meshgrid(x_center_pixel, y_center_pixel, indexing='ij')
        r = np.sqrt(grid_x**2 + grid_y**2)
        print(f"Calculated transverse distances for each pixel, shape: {r.shape}")
        print(f"Transverse distance range: {r.min()} - {r.max()}")
        

        return r


    def radial_column_density_profile(
        self,
        column_density_2d,
        log_range=None,
        log_bins=None,
    ):
        """
        Compute radial column density profile N(r).
        """

        r = self.transverse_distance.to(self.cfg.galaxy.extend_unit).value

        # Define bins
        if log_bins is not None and log_range is not None:
            bins = np.linspace(
                log_range[0],
                log_range[1],
                log_bins + 1
            )
            r=np.log10(r)
        else:
            bins = np.linspace(self.cfg.galaxy.range_transverse[0],
             self.cfg.galaxy.range_transverse[1], 
             self.cfg.galaxy.bins_transverse + 1)

        #how many pixels are in each radial bin
        pixels_per_bin, edges = np.histogram(
            r,
            bins=bins,
            
        )
        weights = column_density_2d.to_physical().value
        total_column_density_per_bin, _ = np.histogram(
            r,
            bins=bins,
            weights=weights
        )

        # Annulus area
        r_outer = edges[1:]
        r_inner = edges[:-1]

        # calculate the average column density in each annulus
        column_density_average = total_column_density_per_bin / pixels_per_bin
        column_density = column_density_average*u.Unit("1/cm**2")

        r_centers = 0.5 * (r_outer + r_inner)*u.Unit(self.cfg.galaxy.extend_unit)
        

        return r_centers, column_density





        
        
        





