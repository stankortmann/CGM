import os

import matplotlib.pyplot as plt
import numpy as np

from .spectrum_io import SpectraReader


class ShortSpectraPlotter:
    def __init__(self, output_dir, constants):
        self.output_dir = output_dir
        self.constants = constants
        amu = constants["amu"]

        self.masses = {
            "Hydrogen": 1.00784 * amu,
            "Helium": 4.002602 * amu,
            "Carbon": 12.0107 * amu,
            "Nitrogen": 14.0067 * amu,
            "Oxygen": 15.999 * amu,
            "Neon": 20.1797 * amu,
            "Magnesium": 24.305 * amu,
            "Silicon": 28.0855 * amu,
            "Sulfur": 32.06 * amu,
            "Calcium": 40.078 * amu,
            "Iron": 55.845 * amu,
}

    @staticmethod
    def safe_name(name):
        return name.replace(" ", "_").replace("/", "_")


    def plot_combined_short_spectrum(self, file_path, nsight, show=True, save_path=None):
        reader = SpectraReader(file_path)
        ions = reader.list_ions(nsight)
        if len(ions) == 0:
            raise ValueError(f"No ions found for LOS_{nsight}")

        ncols = len(ions)
        nrows = 6
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 18), squeeze=False)
        fig.suptitle(f"Short spectrum diagnostics for LOS_{nsight}", fontsize=18)

        for col, (element, ion) in enumerate(ions):
            spectrum = reader.read_ion_spectrum(nsight, element, ion)
            ion_weighted = reader.read_ion_spectrum(nsight, element, ion, weighted_group="Ion-weighted")
            element_weighted = reader.read_ion_spectrum(nsight, element, ion, weighted_group="Element-weighted")

            tau = spectrum["optical_depths"]
            if tau is None:
                continue

            pixel_kms = np.asarray(spectrum["pixel_kms"], dtype=float).reshape(-1)
            if pixel_kms.size == 0:
                raise KeyError(f"pixel_kms not found for LOS_{nsight}/{element}/{ion}")
            x = np.arange(len(tau), dtype=float) * float(pixel_kms[0])
            xlabel = r"Velocity [km/s]"
            transmission = np.exp(-tau)

            element_density = element_weighted["densities"]
            
            ion_density = ion_weighted["densities"]
            tau_weighted_density = spectrum["densities"]
            tau_weighted_number_density = tau_weighted_density / self.masses.get(element, 1.0 * self.constants["amu"])
            tau_weighted_temperature = spectrum["temperatures"]
            tau_weighted_hydrogen_number_density = spectrum["hydrogen_densities"]
            tau_weighted_metallicity = spectrum["metallicities"]

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(element_density > 0, ion_density / element_density, np.nan)
                log_ratio = np.log10(ratio)
                log_ion_density = np.log10(np.where(ion_density > 0, ion_density, np.nan))
                log_element_density = np.log10(np.where(element_density > 0, element_density, np.nan))
                log_tau_weighted_density = np.log10(np.where(tau_weighted_density > 0, tau_weighted_density, np.nan))
                log_tau_weighted_number_density = np.log10(np.where(tau_weighted_number_density.value > 0, tau_weighted_number_density.value, np.nan))
                log_tau_weighted_temperature = np.log10(np.where(tau_weighted_temperature > 0, tau_weighted_temperature, np.nan))
                log_tau_weighted_hydrogen_number_density = np.log10(np.where(tau_weighted_hydrogen_number_density > 0, tau_weighted_hydrogen_number_density, np.nan))

            axes[0, col].plot(x, tau, color="k", lw=1.2)
            axes[0, col].set_title(f"{ion}, log N = {np.log10(spectrum['total_ion_column_density']):.2f}", fontsize=14)
            axes[0, col].set_ylabel(r"$\tau$")
            axes[0, col].set_xlabel(xlabel)
            axes[0, col].set_ylim(0, 2)

            axes[1, col].plot(x, transmission, color="tab:green", lw=1.2)
            axes[1, col].set_ylabel(r"$T=\exp(-\tau)$")
            axes[1, col].set_xlabel(xlabel)
            axes[1, col].set_ylim(0, 1)

            axes[2, col].plot(x, log_tau_weighted_number_density, color="tab:blue", lw=1.2)
            axes[2, col].set_ylabel(r"$\log_{10}(n_{\tau\,weighted})[cm^{-3}]$")
            axes[2, col].set_xlabel(xlabel)
            axes[2, col].set_ylim(-9, -2)


            axes[3, col].plot(x, log_tau_weighted_hydrogen_number_density, color="tab:blue", lw=1.2)
            axes[3, col].set_ylabel(r"$\log_{10}(n_{H\,\tau\,weighted})[cm^{-3}]$")
            axes[3, col].set_xlabel(xlabel)
            axes[3, col].set_ylim(-7, -2)

            axes[4, col].plot(x, log_tau_weighted_temperature, color="tab:red", lw=1.2)
            axes[4, col].set_ylabel(r"$\log_{10}(T_{\tau\,weighted})$")
            axes[4, col].set_xlabel(xlabel)
            axes[4, col].set_ylim(4, 8)

            Z_solar = 0.0134
            axes[5, col].plot(x, tau_weighted_metallicity / Z_solar, color="tab:purple", lw=1.2)
            axes[5, col].set_ylabel(r"$Z_{\tau\,weighted}[Z_{\odot}]$")
            axes[5, col].set_xlabel(xlabel)
            axes[5, col].set_ylim(0.2, 1.5)


            for row in range(nrows):
                axes[row, col].grid(alpha=0.25)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path is None:
            #double path dirname to get the parent directory of the hdf5 file, then add combined_short_spectrum
            file_dir=  os.path.dirname(os.path.dirname(file_path))+"/combined_short_spectrum"
            os.makedirs(file_dir, exist_ok=True)
            save_path = os.path.join(file_dir, f"LOS_{nsight}.png")
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"Saved combined short spectrum plot to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return save_path

    def plot_short_spectrum(self, file_path, nsight, element, ion, save_path=None, show=True, ax=None):
        reader = SpectraReader(file_path)
        spectrum = reader.read_ion_spectrum(nsight, element, ion)
        y = spectrum["optical_depths"]
        if y is None:
            raise KeyError(f"Optical depths not found for LOS_{nsight}/{element}/{ion}")

        pixel_kms = np.asarray(spectrum["pixel_kms"], dtype=float).reshape(-1)
        if pixel_kms.size == 0:
            raise KeyError(f"pixel_kms not found for LOS_{nsight}/{element}/{ion}")
        x = np.arange(len(y), dtype=float) * float(pixel_kms[0])
        xlabel = r"Velocity [km/s]"
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        ax.plot(x, y, color="k", lw=1.4, label=f"{element} {ion}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Optical depth")
        ax.set_title(f"Short spectrum: LOS_{nsight} {element} {ion}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

        if save_path is None:
            save_path = reader.default_plot_path(nsight, element, ion, "short_spectrum")
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved short spectrum plot to: {save_path}")
        if show:
            plt.show()
        return save_path


class LongSpectraPlotter:
    @staticmethod
    def total_tau(outputs):
        total_tau = None
        for ion_data in outputs["Ions"].values():
            tau = np.asarray(ion_data["Optical depths"]["Value"].value)
            if total_tau is None:
                total_tau = np.zeros_like(tau)
            total_tau += tau
        return total_tau

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

            if lambda0 > 0.0 and fvalue > 0.0:
                n_v = 3.768e14 * tau / (fvalue * lambda0)
                n_v = np.asarray(n_v.value if hasattr(n_v, "value") else n_v, dtype=float)
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

    def plot_full_spectrum_keV(self, outputs, out_file):
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


    def plot_long_spectrum(self, file_path, nsight, element, ion, save_path=None, show=True, ax=None):
        reader = SpectraReader(file_path)
        spectrum = reader.read_ion_spectrum(nsight, element, ion)
        y = spectrum["optical_depths"]
        if y is None:
            raise KeyError(f"Optical depths not found for LOS_{nsight}/{element}/{ion}")

        x = spectrum["velocities"]
        xlabel = r"Velocity [km/s]"
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4))

        ax.plot(x, y, color="tab:blue", lw=1.1, label=f"{element} {ion}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Optical depth")
        ax.set_title(f"Long spectrum: LOS_{nsight} {element} {ion}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

        if save_path is None:
            save_path = reader.default_plot_path(nsight, element, ion, "long_spectrum")
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved long spectrum plot to: {save_path}")
        if show:
            plt.show()
        return save_path
