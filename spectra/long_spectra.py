import os
import specwizard
import copy
import matplotlib.pyplot as plt
import numpy as np

from .spectrum_io import LongSpectraWriter, SpectraReader


class long_spectra:
    def __init__(self, wizard, out_dir="/cosma8/data/do012/dc-kort1/CGM/long_spectra"):
        self.wizard = wizard
        self.out_dir = out_dir
        self.output_dir = self.prepare_output_dir()
        self.writer = LongSpectraWriter()
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

    def _run_output_dir(self, coven):
        run_dir = self.output_dir
        os.makedirs(run_dir, exist_ok=True)
        #make numeric subdirectory for each run to avoid overwriting previous runs
        numeric_subdirs = []
        for name in os.listdir(run_dir):
            full_path = os.path.join(run_dir, name)
            if os.path.isdir(full_path) and name.isdigit():
                numeric_subdirs.append(int(name))

        next_index = 0 if len(numeric_subdirs) == 0 else max(numeric_subdirs) + 1
        run_dir = os.path.join(run_dir, str(next_index))
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def run_spectra(
        self,
        add_contaminants=False,
        add_hi_damping_n=None,
        rebin_to_spectrograph=False,
        save_basename="long_spectra",
        save_kev_plot=True,
    ):
        ls_engine = specwizard.LongSpectra(self.wizard)
        ls_engine.paper = True
        ls_engine.random_los = True
        coven, redshifts = ls_engine.create_coven()
        outputs = ls_engine.do_long_spectra(coven)

        run_output_dir = self._run_output_dir(coven)
        projections_list = []
        for wizard in coven:
            snapshot = specwizard.ReadData(wizard=wizard)
            particles = snapshot.read_particles()
            snapshot.header["Cosmo"]["Redshift"] = wizard["z_los"]["Value"]

            sightlineprojection = specwizard.SightLineProjection(wizard)
            projected_los = sightlineprojection.ProjectData(particles)

            wizard_for_od = copy.deepcopy(wizard)
            wizard_for_od.setdefault("ODParams", {})
            wizard_for_od["ODParams"]["VoigtOff"] = True
            cspec = specwizard.ComputeOpticaldepth(wizard_for_od)
            opticaldepth = cspec.MakeAllOpticaldepth(projected_los)

            projections_list.append(
                {
                    "wizard": wizard,
                    "nsight": wizard.get("sightline", {}).get("nsight", 0),
                    "Projection": projected_los,
                    "OpticaldepthWeighted": opticaldepth,
                }
            )

        hdf5_file = self.writer.write_opticaldepth(projections_list, run_output_dir, save_basename)
        self.last_hdf5_file = hdf5_file

        if add_contaminants:
            outputs = ls_engine.add_contaminants(outputs)

        if add_hi_damping_n is not None:
            outputs = ls_engine.add_HI_damping_wings(outputs, n=int(add_hi_damping_n))

        rebinned = None
        if rebin_to_spectrograph:
            rebinned = ls_engine.rebin_to_spectrograph(outputs)

        return {
            "coven": coven,
            "redshifts": redshifts,
            "longspectra": outputs,
            "rebinned": rebinned,
            "hdf5_file": hdf5_file,
        }

    def plot_spectra(self, hdf5_file=None, show=True):
        file_to_plot = hdf5_file or self.last_hdf5_file
        if file_to_plot is None:
            raise ValueError("No HDF5 file available. Run run_spectra() first or provide hdf5_file.")
        reader = SpectraReader(file_to_plot)
        sightlines = reader.list_sightlines()
        if len(sightlines) == 0:
            raise ValueError("No sightlines found in HDF5 file.")

        total_tau = None
        x_axis = None
        x_label = "Velocity [km/s]"

        for sightline in sightlines:
            sightline_id = sightline.replace("LOS_", "")
            for element_name, ion_name in reader.list_ions(sightline_id):
                spectrum = reader.read_ion_spectrum(sightline_id, element_name, ion_name)
                tau = spectrum["optical_depths"]
                if tau is None:
                    continue

                if total_tau is None:
                    total_tau = np.zeros_like(tau, dtype=float)
                    if spectrum["velocities"] is not None:
                        x_axis = spectrum["velocities"]
                        x_label = "Velocity [km/s]"
                    elif spectrum["pixel_kms"] is not None:
                        x_axis = spectrum["pixel_kms"]
                        x_label = "Pixel [km/s]"
                    else:
                        x_axis = np.arange(len(tau))
                        x_label = "Pixel index"

                total_tau += np.asarray(tau, dtype=float)

        if total_tau is None or x_axis is None:
            raise ValueError("Could not assemble a full spectrum from the HDF5 file.")

        transmission = np.exp(-total_tau)
        out_file = os.path.join(os.path.dirname(file_to_plot), "full_spectrum.png")

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(x_axis, transmission, color="k", lw=1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\exp(-\tau_{\rm total})$")
        ax.set_title("Full Transmission Spectrum")
        ax.set_ylim(0.0, 1.02)
        fig.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved full spectrum plot to: {out_file}")
        if show:
            plt.show()
        return out_file

    def run(self):
        return self.run_spectra()

   