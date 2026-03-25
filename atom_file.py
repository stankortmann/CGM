import specwizard

atom_file = specwizard.Atomfile(do_all_elements=True)
name_file = "/cosma8/data/do012/dc-kort1/CGM/atoms/atom_file.hdf5"
atom_file.create_hdf5_from_nist(file_name=name_file, wavelength_low_lim=100.0, wavelength_upper_lim=8500.0)