import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from pathlib import Path

class temperature_density_plotter:
    def __init__(self, density_edges,temperature_edges):
        self.xedges = density_edges
        self.yedges = temperature_edges

    def plot(self,
            density_values,
            density_unit,
            title=None,
            log_scale=True,
            output_path="temperature_density_plot.png"
            ):
            
        fig, ax = plt.subplots(figsize=(7,6))

        if log_scale:
            norm = LogNorm()
        else:
            norm = Normalize()
        # Plot
        mesh = ax.pcolormesh(
            self.xedges,
            self.yedges,
            density_values.T,                # transpose is required for correct orientation
            norm=norm,      # log colour scale (important!)
            shading="auto"
        )

        ax.set_ylabel(r"Log temperature $[K]$")
        ax.set_xlabel(r"Log $n_H/[cm^{-3}]$")

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(density_unit)

        plt.tight_layout()
        plt.title(title)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        return

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class column_density_plotter:

    def __init__(self, x_edges, y_edges,data_unpacker,length_unit="Mpc"):
        self.xedges = x_edges
        self.yedges = y_edges
        self.length_unit=length_unit
        self.output_dir = Path(data_unpacker.output_directory)

    

    def _resolve_name(self, ion=None, element=None):
        if ion is not None:
            return ion
        if element is not None:
            return element
        raise ValueError("Either ion or element must be provided.")


    def plot_xy(
        self,
        column_density_values,
        ion=None,
        element=None,
        log_scale=True,
    ):

        name = self._resolve_name(ion, element)

        fig, ax = plt.subplots(figsize=(7, 6))

        norm = LogNorm() if log_scale else None

        mesh = ax.pcolormesh(
            self.xedges,
            self.yedges,
            column_density_values.T,
            norm=norm,
            shading="auto",
        )

        ax.set_xlabel(rf"x [{self.length_unit}]")
        ax.set_ylabel(rf"y [{self.length_unit}]")
        ax.set_title(f"Column density of {name} in x-y plane")

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(rf"$n_{{{name}}}\,[\mathrm{{cm}}^{{-2}}]$")

        plt.tight_layout()

        file_path = self.output_dir / f"column_density_{name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Finished", file_path)


    def plot_cddf_hist(
        self,
        cddf,
        bin_centers,
        bin_width,
        ion=None,
        element=None,
        range_plot=None,
    ):

        name = self._resolve_name(ion, element)

        plt.figure(figsize=(7, 6))

        plt.bar(bin_centers, cddf, width=bin_width)

        plt.ylabel("CDDF")
        plt.xlabel(rf"$\log_{{10}}(N_{{{name}}}\,[\mathrm{{cm}}^{{-2}}])$")

        plt.title(f"CDDF of {name}")

        if range_plot is not None:
            plt.xlim(range_plot[0], range_plot[1])

        plt.tight_layout()

        file_path = self.output_dir / f"CDDF_{name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Finished", file_path)
    
    def plot_transverse(
        self,
        column_density_values,
        r_centers,
        r_widths,
        ion=None,
        element=None,
        log_scale=False,
        normalize=False
    ):
        """
        Plot 1D radial transverse column density profile.

        Parameters
        ----------
        column_density_values : array
            Column density profile (already in 1/cm^2, passed as values).
        r_centers : array
            Radial bin centers (with length unit already applied).
        r_err : array (optional)
            Error bars in radius (half bin width etc.).
        """

        name = self._resolve_name(ion, element)

        fig, ax = plt.subplots(figsize=(7, 6))

        
        y = np.asarray(column_density_values)
        x = np.asarray(r_centers)


        ax.bar(
            x,
            y,
            width=width,
            align="center",
            alpha=0.7
        )

        ax.set_xlabel(rf"Radius [{self.length_unit}]")
        
        ax.set_ylabel(rf"$N_{{{name}}}\,[\mathrm{{cm}}^{{-2}}]$")

        ax.set_title(f"Radial transverse profile of {name}")

        if log_scale:
            ax.set_yscale("log")

    

        plt.tight_layout()

        file_path = self.output_dir / f"radial_transverse_{name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Finished", file_path)
