import os

import matplotlib.pyplot as plt
import numpy as np
import specwizard
from specwizard import Phys


class long_spectra:
    def __init__(self, wizard, out_dir="/cosma8/data/do012/dc-kort1/CGM/long_spectra"):
        self.wizard = wizard
        self.out_dir = out_dir
        self.constants = Phys.ReadPhys()
        self.output_dir = self._prepare_output_dir()

    @staticmethod
    def _safe_name(name):
        return name.replace(" ", "_").replace("/", "_")

    def _prepare_output_dir(self):
        out_dir = self.out_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            print(f"Warning: Could not create {out_dir}: {exc}")
            print("Falling back to /tmp")
            out_dir = "/tmp/specwizard_output"
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def create_coven(self):
        """Create the long-spectra segment list and corresponding redshifts."""
        ls_engine = specwizard.LongSpectra(self.wizard)
        # Prevent the internal hard-coded "paper" file override path from being used.
        ls_engine.paper = True
        coven, redshifts = ls_engine.create_coven()
        return coven, redshifts

    def _ion_tag(self, ion_key):
        element, ion = ion_key
        return f"{self._safe_name(element)}__{self._safe_name(ion)}"

    def _save_npz(self, outputs, out_file):
        payload = {
            "velocities": np.asarray(outputs["velocities"].value),
            "wavelengths": np.asarray(outputs["wavelengths"].value),
        }

        for ion_key, ion_data in outputs["Ions"].items():
            tag = self._ion_tag(ion_key)
            payload[f"{tag}__tau"] = np.asarray(ion_data["Optical depths"]["Value"].value)
            payload[f"{tag}__vel"] = np.asarray(ion_data["Velocities"]["Value"].value)
            payload[f"{tag}__dens"] = np.asarray(ion_data["Densities"]["Value"].value)
            payload[f"{tag}__temp"] = np.asarray(ion_data["Temperatures"]["Value"].value)
            payload[f"{tag}__lambda0"] = np.asarray([ion_data["lambda0"]])
            payload[f"{tag}__fvalue"] = np.asarray([ion_data["f-value"]])

        np.savez_compressed(out_file, **payload)
        print(f"Saved long-spectra arrays to: {out_file}")

    def _plot_summary(self, outputs, out_file):
        ions = list(outputs["Ions"].keys())
        num_of_ions = len(ions)
        if num_of_ions == 0:
            return

        fig, ax = plt.subplots(num_of_ions, 1, figsize=(20, max(10, 3 * num_of_ions)))
        if num_of_ions == 1:
            ax = [ax]

        velocity = np.asarray(outputs["velocities"].value)

        for i, ion_key in enumerate(ions):
            element, ion_name = ion_key
            tau = np.asarray(outputs["Ions"][ion_key]["Optical depths"]["Value"].value)
            transmission = np.exp(-tau)
            ax[i].plot(velocity, transmission, color="k")
            ax[i].set_title(f"{element} {ion_name}")
            ax[i].set_ylabel(r"$\exp(-\tau)$")
            ax[i].set_xlabel(r"$v\ [{\rm km\ s}^{-1}]$")

        fig.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved long-spectra summary plot to: {out_file}")

    def run(
        self,
        add_contaminants=False,
        add_hi_damping_n=None,
        rebin_to_spectrograph=False,
        save_basename="long_spectra",
    ):
        """
        Build long spectra and optionally apply contaminants/damping/rebinning.

        Returns
        -------
        dict
            Dictionary containing `coven`, `redshifts`, `longspectra`, and optionally `rebinned`.
        """
        ls_engine = specwizard.LongSpectra(self.wizard)
        # Prevent the internal hard-coded "paper" file override path from being used.
        ls_engine.paper = True
        coven, redshifts = ls_engine.create_coven()
        outputs = ls_engine.do_long_spectra(coven)

        if add_contaminants:
            outputs = ls_engine.add_contaminants(outputs)

        if add_hi_damping_n is not None:
            outputs = ls_engine.add_HI_damping_wings(outputs, n=int(add_hi_damping_n))

        rebinned = None
        if rebin_to_spectrograph:
            rebinned = ls_engine.rebin_to_spectrograph(outputs)

        npz_file = os.path.join(self.output_dir, f"{save_basename}.npz")
        self._save_npz(outputs, npz_file)

        summary_file = os.path.join(self.output_dir, f"{save_basename}_transmission.png")
        self._plot_summary(outputs, summary_file)

        return {
            "coven": coven,
            "redshifts": redshifts,
            "longspectra": outputs,
            "rebinned": rebinned,
        }

   