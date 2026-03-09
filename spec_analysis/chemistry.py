from unyt import g, mol, Msun, cm, s, K
import unyt as u
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import h5py


class elements:
        
    @staticmethod
    def particles_per_mass(element):
        """
        Return number of particles per solar mass
        for a given molar mass (with units).
        """
        molar_masses = {
                    "hydrogen": 1.008 * g/mol,
                    "helium": 4.0026 * g/mol,
                    "carbon": 12.011 * g/mol,
                    "nitrogen": 14.007 * g/mol,
                    "oxygen": 15.999 * g/mol,
                    "neon": 20.1797 * g/mol,
                    "magnesium": 24.305 * g/mol,
                    "silicon": 28.085 * g/mol,
                    "iron": 55.845 * g/mol,
                    "strontium": 87.62 * g/mol,
                    "barium": 137.327 * g/mol,
                    "europium": 151.964 * g/mol,
                    }

    
        avogadro_number = 6.02214076e23 / mol  # Avogadro's number in particles/mol
        # Convert moles → particles per gram
        particles = avogadro_number / molar_masses[element] 
        
        return particles
    @staticmethod
    def get_particle_density(element,gas_particles,physical=True):
        """
        Returns the number of particles of a given element in a gas particle.
        """
        element_frac = getattr(gas_particles.element_mass_fractions, element).to_physical() #always physical!!!
        if physical:
            element_mass_densities = gas_particles.densities.to_physical() * element_frac #always physical!!!
        #comoving, almost never used
        else:
            element_mass_densities = gas_particles.densities * element_frac
        element_particles_densities = element_mass_densities * elements.particles_per_mass(element)
        
        return element_particles_densities

        
    @staticmethod
    def get_particle_mass(element,gas_particles):
        element_frac = getattr(gas_particles.element_mass_fractions, element).to_physical() #always physical!!!
        element_mass= gas_particles.masses.to_physical() * element_frac #always physical!!!
        return element_mass


    @staticmethod
    def get_particle_number(element,gas_particles):
        element_frac = getattr(gas_particles.element_mass_fractions, element).to_physical() #always physical!!!
        element_mass= gas_particles.masses.to_physical() * element_frac #always physical!!!
        element_number = element_mass * elements.particles_per_mass(element)
        return element_number


class simulation_constants:
    def __init__(self,resolution):
        self.G = 6.67430e-8 * cm**3 / (g * s**2)  # Gravitational constant in cgs units
        self.c = 29979245800.0 * cm / s  # Speed of light in cgs units
        self.gas_particle_mass = self.get_gas_particle_mass(resolution)  # Gas particle mass in Msun, depends on resolution


    def get_gas_particle_mass(self, resolution):
        gas_mass_dict = {
            5: 2.30e5 * Msun,
            6: 1.84e6 * Msun,
            7: 1.47e7 * Msun,
        }
        return gas_mass_dict[resolution]


#Uses the CHIMES equilibrium table to extract the abundance of a given ion for a given log_Z, log_T, and log_nH_cm3
class chimes:
    def __init__(self, chimes_table_path):
        self.chimes_table_path = chimes_table_path
        self.load_chimes_table()


    def load_chimes_table(self):
        with h5py.File(self.chimes_table_path, "r") as f:
    
            # 4D abundance grid
            #abundances=(N_Temperatures x N_Densities x N_Metallicities x N_species)
            self.abundances = f["Abundances"][:]  
            
            
            self.log_T_table  = f["TableBins/Temperatures"][:]
            self.log_n_H_cm3_table = f["TableBins/Densities"][:]
            self.log_Z_table  = f["TableBins/Metallicities"][:]

        
    def extract_ion_abundance(self,ion,log_Z,log_T,log_n_H_cm3):
        # Load the Chimes table using chimestools
        chimes_dict = {"elec": 0,
                    "HI": 1,
                    "HII": 2,
                    "Hm": 3,
                    "HeI": 4,
                    "HeII": 5,
                    "HeIII": 6,
                    "CI": 7,
                    "CII": 8,
                    "CIII": 9,
                    "CIV": 10,
                    "CV": 11,
                    "CVI": 12,
                    "CVII": 13,
                    "Cm": 14,
                    "NI": 15,
                    "NII": 16,
                    "NIII": 17,
                    "NIV": 18,
                    "NV": 19,
                    "NVI": 20,
                    "NVII": 21,
                    "NVIII": 22,
                    "OI": 23,
                    "OII": 24,
                    "OIII": 25,
                    "OIV": 26,
                    "OV": 27,
                    "OVI": 28,
                    "OVII": 29,
                    "OVIII": 30,
                    "OIX": 31,
                    "Om": 32,
                    "NeI": 33,
                    "NeII": 34,
                    "NeIII": 35,
                    "NeIV": 36,
                    "NeV": 37,
                    "NeVI": 38,
                    "NeVII": 39,
                    "NeVIII": 40,
                    "NeIX": 41,
                    "NeX": 42,
                    "NeXI": 43,
                    "MgI": 44,
                    "MgII": 45,
                    "MgIII": 46,
                    "MgIV": 47,
                    "MgV": 48,
                    "MgVI": 49,
                    "MgVII": 50,
                    "MgVIII": 51,
                    "MgIX": 52,
                    "MgX": 53,
                    "MgXI": 54,
                    "MgXII": 55,
                    "MgXIII": 56,
                    "SiI": 57,
                    "SiII": 58,
                    "SiIII": 59,
                    "SiIV": 60,
                    "SiV": 61,
                    "SiVI": 62,
                    "SiVII": 63,
                    "SiVIII": 64,
                    "SiIX": 65,
                    "SiX": 66,
                    "SiXI": 67,
                    "SiXII": 68,
                    "SiXIII": 69,
                    "SiXIV": 70,
                    "SiXV": 71,
                    "SI": 72,
                    "SII": 73,
                    "SIII": 74,
                    "SIV": 75,
                    "SV": 76,
                    "SVI": 77,
                    "SVII": 78,
                    "SVIII": 79,
                    "SIX": 80,
                    "SX": 81,
                    "SXI": 82,
                    "SXII": 83,
                    "SXIII": 84,
                    "SXIV": 85,
                    "SXV": 86,
                    "SXVI": 87,
                    "SXVII": 88,
                    "CaI": 89,
                    "CaII": 90,
                    "CaIII": 91,
                    "CaIV": 92,
                    "CaV": 93,
                    "CaVI": 94,
                    "CaVII": 95,
                    "CaVIII": 96,
                    "CaIX": 97,
                    "CaX": 98,
                    "CaXI": 99,
                    "CaXII": 100,
                    "CaXIII": 101,
                    "CaXIV": 102,
                    "CaXV": 103,
                    "CaXVI": 104,
                    "CaXVII": 105,
                    "CaXVIII": 106,
                    "CaXIX": 107,
                    "CaXX": 108,
                    "CaXXI": 109,
                    "FeI": 110,
                    "FeII": 111,
                    "FeIII": 112,
                    "FeIV": 113,
                    "FeV": 114,
                    "FeVI": 115,
                    "FeVII": 116,
                    "FeVIII": 117,
                    "FeIX": 118,
                    "FeX": 119,
                    "FeXI": 120,
                    "FeXII": 121,
                    "FeXIII": 122,
                    "FeXIV": 123,
                    "FeXV": 124,
                    "FeXVI": 125,
                    "FeXVII": 126,
                    "FeXVIII": 127,
                    "FeXIX": 128,
                    "FeXX": 129,
                    "FeXXI": 130,
                    "FeXXII": 131,
                    "FeXXIII": 132,
                    "FeXXIV": 133,
                    "FeXXV": 134,
                    "FeXXVI": 135,
                    "FeXXVII": 136,
                    "H2": 137,
                    "H2p": 138,
                    "H3p": 139,
                    "OH": 140,
                    "H2O": 141,
                    "C2": 142,
                    "O2": 143,
                    "HCOp": 144,
                    "CH": 145,
                    "CH2": 146,
                    "CH3p": 147,
                    "CO": 148,
                    "CHp": 149,
                    "CH2p": 150,
                    "OHp": 151,
                    "H2Op": 152,
                    "H3Op": 153,
                    "COp": 154,
                    "HOCp": 155,
                    "O2p": 156}
        ion_index = chimes_dict[ion]
        
        ion_abundance_table = self.abundances[:,:, :, ion_index]

        # Interpolate the abundance for the given log_Z, log_T, and log_n_element_cm3
        interp = RegularGridInterpolator( (self.log_T_table, self.log_n_H_cm3_table, self.log_Z_table), 
                                            ion_abundance_table, 
                                            bounds_error=False, 
                                            fill_value=None)
        #set metallicity ==Nan to lowest metallicity in the table to avoid issues with interpolation when logZ=-inf
        
        
        
        #clip all the input values to be within the bounds of the table to avoid interpolation errors
        log_T = np.clip(log_T,
                        np.min(self.log_T_table),
                        np.max(self.log_T_table))

        log_n_H_cm3 = np.clip(log_n_H_cm3,
                            np.min(self.log_n_H_cm3_table),
                            np.max(self.log_n_H_cm3_table))

        log_Z = np.clip(log_Z,
                        np.min(self.log_Z_table),
                        np.max(self.log_Z_table))
        
        #actual interpolation
        ion_abundance = interp((log_T, log_n_H_cm3, log_Z))
        
        #this is in log10(n_ion/n_element)
        return ion_abundance
        
        