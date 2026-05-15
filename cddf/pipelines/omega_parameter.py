"""
MPI-parallel Omega parameter pipeline using spatial grid + z-slice parallelization.
Uses unwrapper and save_data.py HDF5 logic.
"""

from mpi4py import MPI
import os
import numpy as np
import re
import math
import astropy.units as u
from cddf.chemistry import CHIMES_DICT, elements, chimes
from cddf.unpack_data import unwrapper
from cddf.save_data import omega_saver


def run_omega_parameter(cfg):
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	# Record start time (wall-clock) with MPI timer
	start_time = MPI.Wtime()

	# Validate that size is a perfect square
	grid_size = int(math.sqrt(size))
	if grid_size * grid_size != size:
		if rank == 0:
			print(f"ERROR: Number of cores ({size}) must be a perfect square (4, 16, 25, 36, etc.)")
		comm.Abort()

	# Get number of z-slices from config (must be set via shellscript)
	total_z_slices = getattr(cfg.omega_ion, 'slices', 1)
	if total_z_slices < 1:
		total_z_slices = 1
	slice_start = int(os.environ.get('CD_SLICE_START', '0'))
	slices_per_job = int(os.environ.get('CD_SLICE_COUNT', str(total_z_slices)))
	part_num = int(os.environ.get('JOB_CHUNK_INDEX', '1'))
	if slices_per_job < 1 or slices_per_job == total_z_slices:
		slices_per_job = total_z_slices
		part_num = "total"

	randomize_grid = getattr(cfg.omega_ion, 'randomize_grid', True)
	shuffle_seed = getattr(cfg.omega_ion, 'grid_seed', 0) if randomize_grid else None
	slice_stop = min(slice_start + slices_per_job, total_z_slices)
	if slice_start >= total_z_slices:
		if rank == 0:
			print(
				f"ERROR: slice chunk starts at {slice_start}, "
				f"but only {total_z_slices} total slices were configured"
			)
		comm.Abort()

	if rank == 0:
		print(
			f"Starting Omega parameter calculation: {grid_size}x{grid_size} spatial grid, "
			f"total_slices={total_z_slices}, slices_per_job={slices_per_job}, "
			f"slice_range=[{slice_start}, {slice_stop})"
		)
	
	data_unpacker = unwrapper(cfg)

	# Box size and volume (comoving)
	box_size = data_unpacker.box_size

	# Ensure we have a physical box size and precompute box volume in cm^3
	box_size_phys = box_size.to('Mpc').to_physical()
	box_volume_cm3 = (box_size_phys.to("cm")) ** 3


	# Redshift and cosmology
	redshift = data_unpacker.redshift
	rho_crit_0 = data_unpacker.cosmology.critical_density(0)


	solar_metallicity = 0.0129

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

	# Resolve ion -> element once, before the z-slice loop.
	ion_element_map = {}
	for ion in ion_keys:
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
		ion_element_map[ion] = element_abbrev_dict.get(abbrev)

	

	# Build a one-time reference snapshot on rank 0 so we can determine which
	# elements are actually tracked in the simulation before entering the slice loop.
	if rank == 0:
		reference_region = [[0.0*box_size, 0.0001*box_size], [0.0*box_size, 0.0001*box_size], [0.0*box_size, 0.0001*box_size]]
		reference_snapshot = data_unpacker.load_snapshot(load_region=reference_region)
		reference_gas = reference_snapshot.gas
		available_elements = {
			element_name
			for element_name in element_abbrev_dict.values()
			if hasattr(reference_gas.element_mass_fractions, element_name)
		}
		del reference_snapshot  # free memory
		del reference_gas
		print(f"Available elements in snapshot: {available_elements}")
	else:
		available_elements = None

	available_elements = comm.bcast(available_elements, root=0)

	ions_to_process = [ion for ion in ion_keys if ion_element_map.get(ion) in available_elements]
	# Prepare CHIMES ion fraction table
	chimes_table = chimes(data_unpacker.chimes_table_path, ions_to_cache=ions_to_process)

	if rank == 0:
		print(f"Processing {len(ions_to_process)} ions: {ions_to_process}")

	# Accumulate ion masses across all slices and regions
	accumulated_ion_masses = {ion: 0.0 for ion in ions_to_process}  # in grams

	# Calculate z-slice boundaries
	z_min_comoving = 0.0*box_size
	z_max_comoving = box_size
	z_slice_width = z_max_comoving / total_z_slices

	# Loop over z-slices
	for slice_idx in range(slice_start, slice_stop):
		z_slice_min = z_min_comoving + slice_idx * z_slice_width
		z_slice_max = z_min_comoving + (slice_idx + 1) * z_slice_width

		# Determine this rank's (x, y) grid cell for this slice. If randomize_grid
		# is True we compute a per-slice permutation deterministically from
		# `shuffle_seed` so every rank can derive the same assignment locally.
		if randomize_grid:
			rng = np.random.default_rng(int(shuffle_seed) + int(slice_idx))
			cells = np.arange(size)
			rng.shuffle(cells)
			assigned = int(cells[rank])
			rank_i = assigned // grid_size
			rank_j = assigned % grid_size
		else:
			rank_i = rank // grid_size
			rank_j = rank % grid_size

		x_slice_width = box_size / grid_size
		y_slice_width = box_size / grid_size

		x_min = rank_j * x_slice_width
		x_max = (rank_j + 1) * x_slice_width
		y_min = rank_i * y_slice_width
		y_max = (rank_i + 1) * y_slice_width

		# Create load_region list: [x_min, x_max], [y_min, y_max], [z_slice_min, z_slice_max]
		load_region = [[x_min, x_max], [y_min, y_max], [z_slice_min, z_slice_max]]

		if rank == 0:
			print(f"Slice {slice_idx+1}/{total_z_slices}: All ranks loading their grid regions")

		# Load snapshot with specified region
		snapshot = data_unpacker.load_snapshot(load_region=load_region)
		gas = snapshot.gas

		
		# Gas properties for this region/slice
		log_T = np.log10(gas.temperatures.to_physical().value)
		n_H_cm3 = elements.get_particle_density(
			'hydrogen', gas, physical=True
		).to('1/cm**3').value
		log_n_H_cm3 = np.log10(n_H_cm3)
		Z = gas.metal_mass_fractions.to_physical().value
		metallicity = Z / solar_metallicity
		log_Z = np.log10(np.where(metallicity > 0, metallicity, 1e-40))

		has_species = hasattr(gas, 'species_fractions')

		# Process all ions for this slice/region on this rank
		for ion in ions_to_process:
			element_name = ion_element_map[ion]

			if element_name is None:
				continue

			# Always compute element mass
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

			# Ion mass for this region/slice
			ion_mass = element_mass * ion_frac
			ion_mass_total = ion_mass.sum().to('g').value

			# Accumulate (in grams)
			accumulated_ion_masses[ion] += ion_mass_total

	

	# Gather accumulated masses from all ranks to rank 0
	all_accumulated_masses = comm.gather(accumulated_ion_masses, root=0)
	if rank == 0:
		print(f"Rank {rank}: All slices processed, results gathered from all ranks")

	if rank == 0:
		# Combine results from all ranks
		combined_ion_masses = {ion: 0.0 for ion in ions_to_process}
		for rank_masses in all_accumulated_masses:
			for ion, mass in rank_masses.items():
				combined_ion_masses[ion] += mass

		# Compute Omega parameters
		omega_params = {}
		for ion, total_mass_grams in combined_ion_masses.items():
			avg_density = (total_mass_grams * u.g) / (box_volume_cm3.value *u.cm**3)
			omega = (
				avg_density.to('g/cm**3').value /
				rho_crit_0.to('g/cm**3').value
			)

			print(f"Computed Omega for {ion}: {omega:.3e}")
			omega_params[ion] = omega

		# Save results
		output_dir = data_unpacker.output_directory
		output_filename = f"omega_part{part_num}.hdf5"
		saver = omega_saver(output_dir, redshift, box_size, omega_params, output_filename=output_filename)
		saver.save()

		print(f"Omega parameters saved to {output_dir}")

	# Ensure all ranks reach this point, then report runtime on rank 0
	comm.Barrier()
	end_time = MPI.Wtime()
	elapsed_min = (end_time - start_time) / 60.0
	if rank == 0:
		print(f"Total runtime: {elapsed_min:.2f} minutes")