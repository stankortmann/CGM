import matplotlib.pyplot as plt
import numpy as np


class temperature_density_plotter:
    def __init__(self, density_edges,temperature_edges):
        self.xedges = density_edges
        self.yedges = temperature_edges

    def plot(self,density_values,density_unit,output_path="temperature_density_plot.png"):
        fig, ax = plt.subplots(figsize=(7,6))

        # Plot
        mesh = ax.pcolormesh(
            self.xedges,
            self.yedges,
            self.density_values.T,                # transpose is required for correct orientation
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