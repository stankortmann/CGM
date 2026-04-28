
"""
MPI-parallel Omega parameter pipeline using unwrapper and save_data.py HDF5 logic.
"""

from mpi4py import MPI
import numpy as np
from cddf.chemistry import CHIMES_DICT, elements, chimes
from cddf.unpack_data import unwrapper
from cddf.save_data import omega_saver

def main(cfg):
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	data_unpacker = unwrapper(cfg)
	snapshot = data_unpacker.load_snapshot()
	gas = snapshot.gas

	# Box size and volume (physical)
	box_size = data_unpacker.box_size.to('Mpc').to_physical().value
	box_volume = box_size ** 3  # Mpc^3

	# Redshift and cosmology
	redshift = data_unpacker.redshift
	cosmo = data_unpacker.cosmology
	rho_crit_0 = data_unpacker.cosmology.critical_density(0)

	# Divide particles among ranks
	N = gas.masses.shape[0]
	indices = np.array_split(np.arange(N), size)[rank]

	# Prepare CHIMES ion fraction table
	chimes_table = chimes(data_unpacker.chimes_table_path)

	# Get element, temperature, density, metallicity arrays for local particles
	log_T = np.log10(gas.temperatures[indices].to_physical().value)
	n_H_cm3 = elements.get_particle_density('hydrogen', gas[indices], physical=True).to('1/cm**3').value
	log_n_H_cm3 = np.log10(n_H_cm3)
	solar_metallicity = 0.0129
	Z = gas.metal_mass_fractions[indices].to_physical().value
	metallicity = Z / solar_metallicity
	log_Z = np.log10(np.where(metallicity > 0, metallicity, 1e-40))

	# Prepare output dict
	local_ion_masses = {}
	# Build a dictionary of element abbreviations to full names
	element_abbrev_dict = {
		'H': 'hydrogen',
		'He': 'helium',
		'C': 'carbon',
		'N': 'nitrogen',
		'O': 'oxygen',
		'Ne': 'neon',
		'Mg': 'magnesium',
		'Si': 'silicon',
		'Fe': 'iron',
		'Sr': 'strontium',
		'Ba': 'barium',
		'Eu': 'europium',
		'Ca': 'calcium',
		'S': 'sulfur',
	}

	for ion in CHIMES_DICT:
		# Extract the element abbreviation from the ion name
		abbrev = ''
		for c in ion:
			if c.isupper() and (len(abbrev) == 0 or abbrev[-1].islower()):
				if abbrev:
					break
				abbrev += c
			elif c.islower():
				abbrev += c
			else:
				break
		element = element_abbrev_dict.get(abbrev)
		#continue for elements not in the simulation or not in the CHIMES table
		if element is None or not hasattr(gas.element_mass_fractions, element):
			continue
		# Get element mass for local particles
		element_mass = elements.get_particle_mass(element, gas[indices])
		# Check if the ion fraction is present in the simulation output
		ion_frac = None
		if hasattr(gas, 'species_fractions') and hasattr(gas.species_fractions, ion):
			ion_frac = getattr(gas.species_fractions, ion)[indices].to_physical().value
		else:
			# Get ion fraction from CHIMES
			log_frac = chimes_table.extract_ion_abundance(
				ion=ion,
				log_Z=log_Z,
				log_T=log_T,
				log_n_H_cm3=log_n_H_cm3,
			)
			ion_frac = 10 ** log_frac
		# Ion mass for local particles
		ion_mass = element_mass * ion_frac
		# Sum total ion mass for this rank
		local_ion_masses[ion] = ion_mass.sum().to('g').value

	# Reduce across all ranks
	all_ion_masses = {}
	for ion in local_ion_masses:
		total_mass = comm.reduce(local_ion_masses[ion], op=MPI.SUM, root=0)
		if rank == 0:
			all_ion_masses[ion] = total_mass

	# Only rank 0 writes output
	if rank == 0:
		# Compute average density and Omega for each ion
		omega_params = {}
		for ion, total_mass in all_ion_masses.items():
			avg_density = total_mass / (box_volume.to('cm**3')) 
			omega = avg_density / rho_crit_0
			omega_params[ion] = omega

		output_dir = data_unpacker.output_directory
		saver = omega_saver(output_dir, redshift, box_size, omega_params)
		saver.save()

	comm.Barrier()
