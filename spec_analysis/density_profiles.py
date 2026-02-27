import numpy as np
import unyt as u

#own modules
from spec_analysis import chemistry as chem
from spec_analysis import plot


class column_density_2d:
    """
    Computes 2D column density histograms for elements and their ions.
    Designed for SWIFT + CHIMES workflows.

    We set up the class for a certain element and after that multiple ions can be used of this element
    """

    def __init__(self,cfg,filenames,gas_particles,element):
        self.gas_particles=gas_particles
        self.cfg = cfg
        self.chimes = chem.chimes(filenames.chimes_table_path)

        #We always need the hydrogen number density for the CHIMES table, even if we are not looking at hydrogen
        
        self.n_H_cm3 = chem.elements.get_particle_density(element="hydrogen",
                                                gas_particles=gas_particles,
                                                physical=True).to("1/cm**3")
        
        
        #temperature for each gas particle
        self.temperatures = gas_particles.temperatures.to_physical()
        #metallicity for each gas particle
        self.metallicities= gas_particles.metal_mass_fractions.to_physical()
        #hybrid CHIMES paper solar metallicity (https://richings.bitbucket.io/chimes/user_guide/ChimesData/equilibrium_abundances.html)
        solar_metallicity = 0.0129
        self.metallicities = self.metallicities.value / solar_metallicity #in units of solar metallicity

        self.positions = gas_particles.coordinates.to_physical()

        self.element=element

        #retrieve all relevant parameters of the element
        self.n_element, self.n_element_cm3, self.n_element_column_density=self._column_density_element()


    def _column_density_element(self):
        #number of element particles in each gas particle
        n_element =chem.elements.get_particle_number(self.element, self.gas_particles).value
        
        #element number density for each gas particle
        n_element_cm3 = chem.elements.get_particle_density(element=self.element,
                                                gas_particles=self.gas_particles,
                                                physical=True).to("1/cm**3")
        n_element_hist=np.histogram2d(
            x=self.positions[:,0].to("Mpc").value, #ensure same units for positions
            y=self.positions[:,1].to("Mpc").value, 
            bins=(self.cfg.window.resolution,self.cfg.window.resolution), #squared as to ensure that the total number of bins is resolution^2, as we are doing a 2D histogram 
            range=[[float(self.cfg.window.x[0].to("Mpc").to_physical().value), float(self.cfg.window.x[1].to("Mpc").to_physical().value)],
                    [float(self.cfg.window.y[0].to("Mpc").to_physical().value), float(self.cfg.window.y[1].to("Mpc").to_physical().value)]],
            weights=n_element)
        # Unpack histogram
        n_element_counts, self.xedges, self.yedges = n_element_hist
        #I have ensured that this is all in physical Mpc
        self.pixel_area=(self.xedges[1]-self.xedges[0])*(self.yedges[1]-self.yedges[0])*(u.Mpc)**2
        #convert to column density by dividing by the area of the bin and multiplying by the depth of the box
        n_element_column_density = n_element_counts/(self.pixel_area) 
        #convert to column density in cm^-2 by multiplying by the depth of the box in cm
        n_element_column_density = n_element_column_density.to("1/cm**2").value 
        
        return n_element, n_element_cm3, n_element_column_density
    
    #always first run the corresponding element in column_density_element !!
    #I can make a check of this later on
    def column_density_ion(self,ion):
        
        log_ion_element_fraction=self.chimes.extract_ion_abundance(ion=ion, 
                                        log_Z=np.log10(self.metallicities,
                                            where=self.metallicities>0,
                                            #make sure to avoid logZ=-inf, so set this at -40 which is much lower than the lowest logZ in the table
                                            out=np.full_like(self.metallicities, -40, dtype=float)), 
                                        log_T=np.log10(self.temperatures), 
                                        log_n_H_cm3=np.log10(self.n_H_cm3)
                                        )
        n_ion = self.n_element * 10**log_ion_element_fraction #number of HI particles for each gas particle

        n_ion_hist=np.histogram2d(
            x=self.positions[:,0].to("Mpc").value, #ensure same units for positions
            y=self.positions[:,1].to("Mpc").value, 
            bins=(self.cfg.window.resolution,self.cfg.window.resolution), #squared as to ensure that the total number of bins is resolution^2, as we are doing a 2D histogram 
            range=[[float(self.cfg.window.x[0].to("Mpc").to_physical().value), float(self.cfg.window.x[1].to("Mpc").to_physical().value)],
                [float(self.cfg.window.y[0].to("Mpc").to_physical().value), float(self.cfg.window.y[1].to("Mpc").to_physical().value)]],
            weights=n_ion)
        # Unpack histogram
        n_ion_counts, _, _ = n_ion_hist
        #convert to column density by dividing by the area of the bin and multiplying by the depth of the box
        n_ion_column_density = n_ion_counts/(self.pixel_area) 
        #convert to column density in cm^-2 by multiplying by the depth of the box in cm
        n_ion_column_density = n_ion_column_density.to("1/cm**2")

        return n_ion_column_density
    
    def column_density_distribution_function(self,ion,log_column_density_range=None,n_bins=100,normalize=True):
        
        #flatten the 2D array to just count the values
        n_ion_column_density=self.column_density_ion(ion=ion).flatten()
        #extra safety measure and now only get the value
        n_ion_column_density=n_ion_column_density.to("1/cm**2").value
        #ensure that pixels without particles are filtered out
        n_ion_column_density=n_ion_column_density[n_ion_column_density>0]
        
        log_n_ion_column_density=np.log10(n_ion_column_density)
        if log_column_density_range == None:
            #maybe clip the ranges here
            min_log_cd=np.min(log_n_ion_column_density)
            max_log_cd=np.max(log_n_ion_column_density)
            log_column_density_range = [min_log_cd,max_log_cd]
        
        log_bins=np.linspace(start=log_column_density_range[0],stop=log_column_density_range[1],num=n_bins)
        cddf, edges = np.histogram(log_n_ion_column_density, bins=log_bins,density=normalize)

        dlog_n_ion_column_density = edges[1] - edges[0]
        log_bin_centers = 0.5 * (edges[1:] + edges[:-1])
        #for now only the CDDF and the log_bins are returned to plot the histogram inside the plotter class
        return cddf,log_bins





        
        
        





