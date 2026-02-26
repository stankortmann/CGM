import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


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

class column_density_plotter:
    def __init__(self, x_edges,y_edges):
        self.xedges = x_edges
        self.yedges = y_edges
        self.redges = None

    def plot_xy(self,
                column_density_values,
                column_density_unit,
                title=None, 
                log_scale=True,
                output_path="column_density_plot.png"
                ):

        fig, ax = plt.subplots(figsize=(7,6))

        # Plot
        mesh = ax.pcolormesh(
            self.xedges,
            self.yedges,
            column_density_values.T,                # transpose is required for correct orientation
            norm=LogNorm(),      # log colour scale (important!)
            shading="auto"
        )

        ax.set_xlabel(r"x [Mpc]")
        ax.set_ylabel(r"y [Mpc]")

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(column_density_unit)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_r(self,
            column_density_values,
            column_density_unit,
            title=None,
            log_scale=False,
            output_path="column_density_plot.png",
            redges=None
            ): 

        fig, ax = plt.subplots(figsize=(7,6))

        # Plot
        mesh = ax.pcolormesh(
            self.xedges,
            self.yedges,
            density_values.T,                # transpose is required for correct orientation
            norm=LogNorm(),      # log colour scale (important!)
            shading="auto"
        )

        ax.set_ylabel(r"Log temperature $[K]$")
        ax.set_xlabel(r"Log $n_H/[cm^{-3}]$")

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label(density_unit)

        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

