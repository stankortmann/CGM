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





class column_density_plotter:

    def __init__(self, cfg, data_unpacker, x_edges, y_edges,cfg_plot=None):

        self.cfg = cfg
        self.data_unpacker = data_unpacker
        if cfg_plot is not None:
            self.cfg_plot = cfg_plot
        self.length_unit = self.cfg.column_density.length_unit_2d

        self.xedges = x_edges.to(self.length_unit).value
        self.yedges = y_edges.to(self.length_unit).value

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
        ax=None,
        save=True
    ):

        name = self._resolve_name(ion, element)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
            created_fig = True

        vlower = self.cfg.column_density.log_range_2d[0]
        vhigher = self.cfg.column_density.log_range_2d[1]

        norm = LogNorm(vmin=vlower, vmax=vhigher) if log_scale else None

        mesh = ax.imshow(
            column_density_values.T,
            origin="lower",
            extent=[
                self.xedges[0], self.xedges[-1],
                self.yedges[0], self.yedges[-1],
            ],
            norm=norm,
            aspect="auto",
        )

        ax.set_xlabel(rf"x [{self.length_unit}]")
        ax.set_ylabel(rf"y [{self.length_unit}]")
        ax.set_title(f"Column density of {name}")

        if created_fig:
            cbar = plt.colorbar(mesh, ax=ax)
            cbar.set_label(rf"$N_{{{name}}}\,[\mathrm{{cm}}^{{-2}}]$")

        if created_fig and save:

            plt.tight_layout()

            cd_dir = self.output_dir / "column_density"
            cd_dir.mkdir(parents=True, exist_ok=True)

            file_path = cd_dir / f"{name}.png"

            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            print("Finished", file_path)
            return

        return ax

    def plot_cddf_hist(
        self,
        cddf,
        bin_centers,
        bin_width,
        ion=None,
        element=None,
        log_scale=True,
        ax=None,
        label=None,
        save=True
    ):

        name = self._resolve_name(ion, element)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
            created_fig = True

        cddf = np.asarray(cddf, dtype=float)

        if log_scale:

            log_cddf = np.full_like(cddf, np.nan)

            mask = cddf > 0
            log_cddf[mask] = np.log10(cddf[mask])

            ax.scatter(bin_centers, log_cddf, label=label)

            ax.set_ylabel(
                rf"$\log_{{10}}f(N_{{{name}}}) = "
                rf"\log_{{10}}\frac{{d^2 n}}{{dN_{{{name}}}\,d\chi}}$"
            )

        else:

            ax.bar(bin_centers, cddf, width=bin_width, label=label)

            ax.set_ylabel(
                rf"$f(N_{{{name}}}) = \frac{{d^2 n}}{{dN_{{{name}}}\,d\chi}}$"
            )

        ax.set_xlabel(rf"$\log_{{10}}(N_{{{name}}})$")

        if label is not None:
            ax.legend()

        if created_fig and save:

            plt.tight_layout()

            cddf_dir = self.output_dir / "CDDF"
            cddf_dir.mkdir(parents=True, exist_ok=True)

            file_path = cddf_dir / f"{name}.png"

            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            print("Finished", file_path)
            return

        return ax