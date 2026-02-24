from unyt import g, mol, Msun, cm, s, K
import unyt as u
import numpy as np


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