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

    def save_npz(self, outputs, out_file):
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

    def plot_summary(self, outputs, out_file):
        ions = list(outputs["Ions"].keys())
        num_of_ions = len(ions)
        if num_of_ions == 0:
            return

        fig, ax = plt.subplots(2 * num_of_ions, 1, figsize=(20, max(12, 5 * num_of_ions)), sharex=True)
        if 2 * num_of_ions == 1:
            ax = [ax]

        velocity = np.asarray(outputs["velocities"].value)

        for i, ion_key in enumerate(ions):
            element, ion_name = ion_key
            tau = np.asarray(outputs["Ions"][ion_key]["Optical depths"]["Value"].value, dtype=float)
            transmission = np.exp(-tau)

            lambda0 = outputs["Ions"][ion_key]["lambda0"]
            fvalue = outputs["Ions"][ion_key]["f-value"]

            # Apparent column density per velocity bin in cm^-2 (km/s)^-1.
            if lambda0 > 0.0 and fvalue > 0.0:
                n_v = 3.768e14 * tau / (fvalue * lambda0)
                n_v = np.asarray(n_v.value if hasattr(n_v, 'value') else n_v, dtype=float)
            else:
                n_v = np.full_like(tau, np.nan, dtype=float)

            n_v[n_v <= 0.0] = np.nan

            upper = 2 * i
            lower = upper + 1

            ax[upper].plot(velocity, transmission, color="k")
            ax[upper].set_title(f"{element} {ion_name}")
            ax[upper].set_ylabel(r"$\exp(-\tau)$")

            ax[lower].plot(velocity, n_v, color="tab:blue")
            ax[lower].set_yscale("log")
            ax[lower].set_ylabel(r"$N_{\rm a}(v)$")
            ax[lower].set_xlabel(r"$v\ [{\rm km\ s}^{-1}]$")

        fig.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved long-spectra summary plot to: {out_file}")

    def total_tau(self, outputs):
        total_tau = None
        for ion_data in outputs["Ions"].values():
            tau = np.asarray(ion_data["Optical depths"]["Value"].value)
            if total_tau is None:
                total_tau = np.zeros_like(tau)
            total_tau += tau
        return total_tau

    def plot_full_spectrum_keV(self, outputs, out_file):
        # Energy conversion: E[keV] = 12.398419843320026 / lambda[Angstrom]
        wavelength_angstrom = np.asarray(outputs["wavelengths"].value)
        total_tau = self.total_tau(outputs)
        if total_tau is None:
            return

        transmission = np.exp(-total_tau)
        energy_keV = 12.398419843320026 / wavelength_angstrom

        sort_idx = np.argsort(energy_keV)
        energy_keV = energy_keV[sort_idx]
        transmission = transmission[sort_idx]

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(energy_keV, transmission, color="k", lw=1.0)
        ax.set_xlabel(r"$E\ [{\rm keV}]$")
        ax.set_ylabel(r"$\exp(-\tau_{\rm total})$")
        ax.set_title("Full Transmission Spectrum")
        ax.set_ylim(0.0, 1.02)
        fig.tight_layout()
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved full keV transmission plot to: {out_file}")

    def column_densities(self, coven):
        if len(coven) == 0:
            print("No sightlines found in coven; nothing to print.")
            return

        print("Column densities by sightline and ion:")
        for wizard in coven:
            los_file = wizard.get("snapshot_params", {}).get("file", "unknown")
            nsight = wizard.get("sightline", {}).get("nsight", "unknown")
            z_los = wizard.get("z_los", {}).get("Value", "unknown")

            snapshot = specwizard.ReadData(wizard=wizard)
            particles = snapshot.read_particles()
            snapshot.header["Cosmo"]["Redshift"] = wizard["z_los"]["Value"]

            sightlineprojection = specwizard.SightLineProjection(wizard)
            projected_los = sightlineprojection.ProjectData(particles)

            cspec = specwizard.ComputeOpticaldepth(wizard)
            opticaldepth = cspec.MakeAllOpticaldepth(projected_los)

            print(f"  file={los_file} nsight={nsight} z_los={z_los}")
            for ion in wizard.get("ionparams", {}).get("Ions", []):
                try:
                    nion = opticaldepth[ion]["TotalIonColumnDensity"]["Value"]
                    nion_value = float(np.asarray(nion))
                    log_nion = np.log10(nion_value) if nion_value > 0 else -np.inf
                    print(f"    {ion[0]} {ion[1]}: log10(N/cm^2) = {log_nion:.4f}")
                except Exception:
                    print(f"    {ion[0]} {ion[1]}: column density unavailable")

    def run(
        self,
        add_contaminants=False,
        add_hi_damping_n=None,
        rebin_to_spectrograph=False,
        save_basename="long_spectra",
        save_kev_plot=True,
        print_column_densities=False,
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

        if print_column_densities:
            self.column_densities(coven)

        run_output_dir = self._run_output_dir(coven)

        if add_contaminants:
            outputs = ls_engine.add_contaminants(outputs)

        if add_hi_damping_n is not None:
            outputs = ls_engine.add_HI_damping_wings(outputs, n=int(add_hi_damping_n))

        rebinned = None
        if rebin_to_spectrograph:
            rebinned = ls_engine.rebin_to_spectrograph(outputs)

        npz_file = os.path.join(run_output_dir, f"{save_basename}.npz")
        self.save_npz(outputs, npz_file)

        summary_file = os.path.join(run_output_dir, f"{save_basename}_transmission.png")
        self.plot_summary(outputs, summary_file)

        if save_kev_plot:
            keV_file = os.path.join(run_output_dir, f"{save_basename}_full_keV_transmission.png")
            self.plot_full_spectrum_keV(outputs, keV_file)

        return {
            "coven": coven,
            "redshifts": redshifts,
            "longspectra": outputs,
            "rebinned": rebinned,
        }

   