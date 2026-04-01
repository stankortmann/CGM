import specwizard
import os

element_list = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Fe"]
atom_file = specwizard.Atomfile(elements_to_do=element_list)
name_file = "/cosma8/data/do012/dc-kort1/CGM/atoms/atom_file_high_energy.hdf5"

# Ensure the output directory exists
os.makedirs(os.path.dirname(name_file), exist_ok=True)

# Create the HDF5 file with verbose output to track progress
atom_file.create_hdf5_from_nist(
    file_name=name_file,
    wavelength_low_lim=1.0,
    wavelength_upper_lim=8500.0,
    verbose=True
)

print(f"Atom file created successfully at: {name_file}")