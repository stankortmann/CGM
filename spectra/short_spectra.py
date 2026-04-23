import os
import specwizard
from specwizard import Phys
import glob
import h5py
import matplotlib.pyplot as plt
import re
import numpy as np
from .spectrum_io import ShortSpectraWriter, SpectraReader
from .spectrum_plotting import ShortSpectraPlotter


class short_spectra:

        
    def __init__(self, wizard, out_dir="/cosma8/data/do012/dc-kort1/CGM/short_spectra"):
        self.wizard = wizard
        self.out_dir = wizard.get("Output", {}).get("directory", out_dir)
        self.output_dir = self.prepare_output_dir()
        self.writer = ShortSpectraWriter(self.output_dir)
        self.plotter = ShortSpectraPlotter(self.output_dir, Phys.ReadPhys())
        self.last_hdf5_file = None

    def prepare_output_dir(self):
        out_dir = self.out_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            print(f"Warning: Could not create {out_dir}: {exc}")
            print("Falling back to /tmp")
            out_dir = "/tmp/specwizard_output"
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def run_spectra(self, wizard=None):
        if wizard is None:
            wizard = self.wizard
        optical_depth, projected_data, snap_data = specwizard.GenerateShortSpectra(Wizard=wizard)
        hdf5_file = self.writer.write_opticaldepth(wizard, projected_data, optical_depth)
        self.last_hdf5_file = hdf5_file
        print(f"Saved short-spectra optical-depth hdf5 to: {hdf5_file}")
        return {"hdf5_file": hdf5_file}

    def plot_spectra(self, hdf5_file=None, show=True):
        file_to_plot = hdf5_file or self.last_hdf5_file
        if file_to_plot is None:
            raise ValueError("No HDF5 file available. Run run_spectra() first or provide hdf5_file.")

        reader = SpectraReader(file_to_plot)
        plotted = []
        for sightline in reader.list_sightlines():
            sightline_id = sightline.replace("LOS_", "")
            plotted.append(self.plotter.plot_combined_short_spectrum(file_path=file_to_plot, nsight=sightline_id, show=show))
        return plotted

    def run(self):
        return self.run_spectra(self.wizard)
    
    def inspect_los(self, directory):
            

        # Find all .hdf5 files in the directory, data is expected to be in a subdirectory called 'hdf5'
        data_directory = os.path.join(directory, 'hdf5')
        hdf5_files = sorted(glob.glob(os.path.join(data_directory, '*.hdf5')))
        if not hdf5_files:
            print(f"No HDF5 files found in {directory}")
            return

        # Map: los_number -> {ion: column_density}
        los_data = {}
        all_ions_sets = []
        los_numbers = []
        los_pattern = re.compile(r'los_(\d+)\.hdf5', re.IGNORECASE)

        for file in hdf5_files:
            m = los_pattern.search(os.path.basename(file))
            if not m:
                continue
            los_num = m.group(1)
            los_numbers.append(los_num)
            reader = SpectraReader(file)
            # Assume only one LOS per file, get its id
            sightlines = reader.list_sightlines()
            if not sightlines:
                continue
            nsight = sightlines[0].replace('LOS_', '')
            ions = reader.list_ions(nsight)
            all_ions_sets.append(set(ions))
            los_data[los_num] = {}
            for element, ion in ions:
                spectrum = reader.read_ion_spectrum(nsight, element, ion)
                col_density = spectrum.get('total_ion_column_density')
                if col_density is not None:
                    # If array, take sum or first value
                    if isinstance(col_density, np.ndarray):
                        col_density = np.sum(col_density)
                    los_data[los_num][(element, ion)] = col_density

        # Find ions present in all files
        if not all_ions_sets:
            print("No ions found in any file.")
            return
        common_ions = set.intersection(*all_ions_sets)
        if not common_ions:
            print("No common ions found across all files.")
            return

        los_numbers_sorted = sorted(los_data.keys(), key=lambda x: int(x))

        for element, ion in sorted(common_ions):
            x = []
            y = []
            for los_num in los_numbers_sorted:
                val = los_data[los_num].get((element, ion))
                if val is not None and val > 0:
                    col_density = np.log10(val)
                else:
                    col_density = np.nan
                x.append(int(los_num))
                y.append(col_density)
            if x and y:
                plt.figure()
                plt.scatter(x, y, marker='o', linestyle='-')
                plt.xlabel('LOS number')
                plt.ylabel(f'log(N) [{ion}]')
                plt.title(f'Column density for {ion} across LOS')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                # Save plot as los_<element>_<ion>.png in the directory
                safe_element = str(element).replace(' ', '_').replace('/', '_')
                safe_ion = str(ion).replace(' ', '_').replace('/', '_')
                plot_filename = f"{safe_ion}.png"
                plot_directory = os.path.join(directory, 'all_los_column_densities')
                os.makedirs(plot_directory, exist_ok=True)
                plot_path = os.path.join(plot_directory, plot_filename)
                plt.savefig(plot_path)
                print(f"Saved column density query for all LOS in the input directory for {ion} to {plot_path}")
                
                
                cutoff_cd=15
                
                print(f"Line-of-sights with log(N) for {ion} higher than {cutoff_cd} detected. These are: ")
                for i, col in enumerate(y):
                    if col > cutoff_cd:
                        print(f"  LOS {los_numbers_sorted[i]} with log(N) = {col:.2f}")
                plt.close()