import os

import matplotlib.pyplot as plt
import numpy as np
import specwizard
from specwizard import Phys


class short_spectra:
    def __init__(self, wizard, out_dir="/cosma8/data/do012/dc-kort1/CGM/short_spectra"):
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

    def single_sightline(self, wizard):
        optical_depth, projected_data, snap_data = specwizard.GenerateShortSpectra(Wizard=wizard)

        fontsize = 20
        positions = snap_data["Particles"]["Positions"]
        densities = snap_data["Particles"]["Densities"]
        elementnames = wizard["ionparams"]["Ions"]

        ix = wizard["sightline"]["x-axis"]
        iy = wizard["sightline"]["y-axis"]
        iz = wizard["sightline"]["z-axis"]
        _ = positions["Value"][:, iz]
        _ = densities["Value"]
        _ = ix, iy

        pixz = projected_data["pixel"]["Value"] * np.arange(projected_data["npix"])

        los_file = os.path.basename(wizard.get("snapshot_params", {}).get("file", "unknown_los"))
        los_file_tag = self._safe_name(os.path.splitext(los_file)[0])
        los_num = wizard.get("sightline", {}).get("nsight", "unknown")
        los_dir = os.path.join(self.output_dir, los_file_tag, f"los_{los_num}")
        os.makedirs(los_dir, exist_ok=True)
        print("printing the column density for the ions")
        for element, ion in elementnames:
            od = optical_depth[(element, ion)]["Optical depths"]["Value"]
            nion = optical_depth[(element, ion)]["TotalIonColumnDensity"]["Value"]
            print(element, ion, np.log10(nion))
            element_density = projected_data["Element-weighted"][element]["Densities"]["Value"]
            ion_density = projected_data["Ion-weighted"][ion]["Densities"]["Value"]
            ion_mass = wizard["ionparams"]["transitionparams"][ion]["Mass"] * self.constants["amu"]
            ion_number_density = (ion_density / ion_mass).in_cgs()
            tau_weighted_particle_density = (optical_depth[(element, ion)]["Densities"]["Value"] / ion_mass).in_cgs()

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(element_density > 0, ion_density / element_density, np.nan)
                log_ratio = np.log10(ratio)
                log_ion_number_density = np.log10(np.where(ion_number_density.value > 0, ion_number_density.value, np.nan))
                log_tau_weighted_particle_density = np.log10(
                    np.where(tau_weighted_particle_density.value > 0, tau_weighted_particle_density.value, np.nan)
                )
            redshift = str(round(wizard["Header"]["Cosmo"]["Redshift"], 2))
            fig, ax = plt.subplots(5, 1, figsize=(20, 20))

            ax[0].plot(od, color="k")
            ax[0].set_title(rf"{element} {ion} at redshift {redshift} with Log N [$cm^{{-2}}$] = {np.log10(nion):.2f}", fontsize=fontsize)
            ax[0].set_ylabel(r"$\tau$", fontsize=fontsize)
            ax[0].set_xlabel(r"Velocity [km/s]", fontsize=fontsize)

            transmission = np.exp(-od)
            ax[1].plot(transmission, color="tab:green")
            ax[1].set_ylabel(rf"$T=\exp(-\tau)$", fontsize=fontsize)
            ax[1].set_xlabel(r"Velocity [km/s]", fontsize=fontsize)

            ax[2].plot(pixz, log_ratio, color="tab:blue")
            ax[2].set_ylabel(rf"$\log_{{10}}(n_{{ion}}/n_{{element}})$", fontsize=fontsize)
            ax[2].set_xlabel(r"$z$ [Mpc]", fontsize=fontsize)

            ax[3].plot(pixz, log_ion_number_density, color="tab:purple")
            ax[3].set_ylabel(r"$\log_{10}(n_{\rm ion}\,[{\rm cm}^{-3}])$", fontsize=fontsize)
            ax[3].set_xlabel(r"$z$ [Mpc]", fontsize=fontsize)

            ax[4].plot(pixz, log_tau_weighted_particle_density, color="tab:red")
            ax[4].set_ylabel(r"$\log_{10}(n_{\rm ion,\tau}\,[{\rm cm}^{-3}])$", fontsize=fontsize)
            ax[4].set_xlabel(r"$z$ [Mpc]", fontsize=fontsize)

            fig.tight_layout()

            ion_tag = self._safe_name(ion)
            element_tag = self._safe_name(element)
            out_file = os.path.join(los_dir, f"individual/{element_tag}/{ion_tag}.png")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            fig.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved plot to: {out_file}")

        num_of_ions = len(elementnames)
        fig, ax = plt.subplots(num_of_ions, 1, figsize=(20, max(10, 3 * num_of_ions)))

        if num_of_ions == 1:
            ax = [ax]

        for i, (element, ion_name) in enumerate(elementnames):
            try:
                od = optical_depth[(element, ion_name)]["Optical depths"]["Value"]
            except KeyError:
                continue

            transmission = np.exp(-od)
            ax[i].plot(transmission, color="k")
            ax[i].set_title(f"{element} {ion_name}")
            ax[i].set_ylabel(r"$\exp(-\tau)$")
            ax[i].set_xlabel(r"$v\ [{\rm km\ s}^{-1}]$")

        fig.tight_layout()
        transmission_file = os.path.join(los_dir, "all_ions_transmission.png")
        fig.savefig(transmission_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(num_of_ions, 1, figsize=(20, max(10, 3 * num_of_ions)))
        for i, (element, ion_name) in enumerate(elementnames):
            try:
                od = optical_depth[(element, ion_name)]["Optical depths"]["Value"]
            except KeyError:
                continue

            ax[i].plot(od, color="k")
            ax[i].set_title(f"{element} {ion_name}")
            ax[i].set_ylabel(r"$\tau$")
            ax[i].set_xlabel(r"$v\ [{\rm km\ s}^{-1}]$")

        fig.tight_layout()
        tau_file = os.path.join(los_dir, "all_ions_tau.png")
        fig.savefig(tau_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plot to: {tau_file}")

    def run(self):
        self.single_sightline(self.wizard)