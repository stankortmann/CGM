import os
import specwizard
from specwizard import Phys

from .spectrum_io import ShortSpectraWriter, SpectraReader
from .spectrum_plotting import ShortSpectraPlotter


class short_spectra:
    def __init__(self, wizard, out_dir="/cosma8/data/do012/dc-kort1/CGM/short_spectra"):
        self.wizard = wizard
        self.out_dir = out_dir
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