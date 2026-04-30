"""
MPI-parallel Omega parameter pipeline using unwrapper and save_data.py HDF5 logic.
"""

from mpi4py import MPI
import numpy as np
import re
from cddf.chemistry import CHIMES_DICT, elements, chimes
from cddf.unpack_data import unwrapper
from cddf.save_data import omega_saver


def run_omega_parameter(cfg):
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	data_unpacker = unwrapper(cfg)
	snapshot = data_unpacker.load_snapshot()
	gas = snapshot.gas
	if rank == 0:
		print(f"The elements tracked in the simulation are: {gas.element_mass_fractions}")

	# Box size and volume (physical)
	box_size = data_unpacker.box_size.to('Mpc').to_physical()
	box_volume = box_size ** 3  # with units

	# Redshift and cosmology
	redshift = data_unpacker.redshift
	rho_crit_0 = data_unpacker.cosmology.critical_density(0)

	# Prepare CHIMES ion fraction table
	chimes_table = chimes(data_unpacker.chimes_table_path)

	# Gas properties
	log_T = np.log10(gas.temperatures.to_physical().value)
	n_H_cm3 = elements.get_particle_density(
		'hydrogen', gas, physical=True
	).to('1/cm**3').value
	log_n_H_cm3 = np.log10(n_H_cm3)

	solar_metallicity = 0.0129
	Z = gas.metal_mass_fractions.to_physical().value
	metallicity = Z / solar_metallicity
	log_Z = np.log10(np.where(metallicity > 0, metallicity, 1e-40))

	# Element mapping
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

	

	def is_ion(name):
		return re.fullmatch(r"[A-Z][a-z]?(?:[IVXLCDM]+)$", name) is not None

	ion_keys = sorted([k for k in CHIMES_DICT.keys() if is_ion(k)])

	# Split work across MPI ranks
	number_of_ions = len(ion_keys)
	ions_per_rank = (number_of_ions + size - 1) // size
	start_ion = rank * ions_per_rank
	end_ion = min(start_ion + ions_per_rank, number_of_ions)

	local_ion_masses = {}

	has_species = hasattr(gas, 'species_fractions')


	for ion in ion_keys[start_ion:end_ion]:
		print(f"Rank {rank} processing ion: {ion}")

		# Extract element abbreviation
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

		element_name = element_abbrev_dict.get(abbrev)

		if element_name is None:
			print(f"Rank {rank} skipping ion {ion} (unknown element: {abbrev})")
			continue

		if not hasattr(gas.element_mass_fractions, element_name):
			print(f"Rank {rank} skipping ion {ion} (element not in simulation)")
			continue

		# Always compute element mass (safe & simple)
		element_mass = elements.get_particle_mass(element_name, gas)

		# Ion fraction
		if has_species and hasattr(gas.species_fractions, ion):
			ion_frac = getattr(gas.species_fractions, ion).to_physical().value
		else:
			log_frac = chimes_table.extract_ion_abundance(
				ion=ion,
				log_Z=log_Z,
				log_T=log_T,
				log_n_H_cm3=log_n_H_cm3,
			)
			ion_frac = 10 ** log_frac

		# Ion mass
		ion_mass = element_mass * ion_frac

		# Store summed mass
		local_ion_masses[ion] = ion_mass.sum().to('g')

	# Compute Omega parameters
	omega_params = {}
	for ion, ion_mass in local_ion_masses.items():
		avg_density = ion_mass / box_volume.to('cm**3')
		omega = (
			avg_density.to('g/cm**3').value /
			rho_crit_0.to('g/cm**3').value
		)

		print(f"Rank {rank} computed Omega for {ion}: {omega:.3e}")
		omega_params[ion] = omega

	# Gather results
	all_omega_params = comm.gather(omega_params, root=0)

	if rank == 0:
		combined_omega_params = {}
		for d in all_omega_params:
			combined_omega_params.update(d)

		output_dir = data_unpacker.output_directory
		saver = omega_saver(output_dir, redshift, box_size, combined_omega_params)
		saver.save()

	comm.Barrier()