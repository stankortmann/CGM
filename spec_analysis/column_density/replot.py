# spec_analysis/column_density/replot.py

import matplotlib.pyplot as plt
from pathlib import Path
from spec_analysis.unpack_data import single_cd, unwrapper
from spec_analysis import plot  # your column_density_plotter class
from spec_analysis.data_structure.plot import plot_config
import numpy as np


def get_label(cd_data,data_unpacker,selection_criterium):

    if selection_criterium == "box_size":
        return f"Box size: {cd_data.cfg.simulation.box_size:.1f} Mpc"
    elif selection_criterium == "redshift":
        return f"z= {data_unpacker.redshift:.2f}"
    elif selection_criterium == "scale_factor":
        return f"a= {data_unpacker.scale_factor:.3f}"
    elif selection_criterium == "particle_resolution":
        return f"m: {cd_data.cfg.simulation.resolution}"
    elif selection_criterium == "pixel_resolution":
        return rf"Pixel number: ${cd_data.cfg.window.resolution}^2$"
    elif selection_criterium == "simulation_name":
        return f"L{cd_data.cfg.simulation.box_length:03d}_m{cd_data.cfg.simulation.resolution}_{cd_data.cfg.simulation.name}"
    #can add more criteria here as needed
    else:
        raise ValueError(f"Unknown label criterion: {selection_criterium}")


def plot_eagle_cddf(ax, ion, cfg_plot):
    """
    Overlay EAGLE CDDF data if cfg_plot.plot_eagle is True
    and the corresponding CSV file exists.
    """
    if not getattr(cfg_plot, "plot_eagle", False):
        return ax

    file_path = cfg_plot.eagle_cddf_directory / f"{ion}.csv"

    if not file_path.exists():
        print(f"EAGLE file not found for {ion}, skipping.")
        return ax

    data = np.loadtxt(file_path, delimiter=",")

    ax.scatter(
        data[:, 0],
        data[:, 1],
        label="EAGLE",
        marker="o",
        s=20,
        color="black",
        zorder=5
    )

    return ax




def run_single(cfg_plot: plot_config):

    data_file = Path(cfg_plot.data_files[0])

    cd_data = single_cd(data_file)
    data_unpacker = unwrapper(cd_data.cfg)

    plotter = plot.column_density_plotter(
        cfg=cd_data.cfg,
        data_unpacker=data_unpacker,
        x_edges=cd_data.xedges,
        y_edges=cd_data.yedges,
        cfg_plot=cfg_plot
    )
    # -------------------------
    # Element XY column density map
    # -------------------------

    plotter.plot_xy(
        column_density_values=cd_data.element_cd,
        element=cd_data.element_name,
        log_scale=True
    )
    output_dir = Path(data_unpacker.output_directory)
    
    
    # -------------------------
    # Element CDDF
    # -------------------------

    plotter.plot_cddf_hist(
        cddf=cd_data.element_cddf,
        bin_centers=cd_data.element_bin_centers,
        bin_width=cd_data.element_bin_width,
        element=cd_data.element_name,
        log_scale=True
    )


    # -------------------------
    # Ion CDDFs
    # -------------------------

    ions_to_plot = cd_data.cfg.chemistry.ion

    for ion in ions_to_plot:

        if ion not in cd_data.ions:
            print(f"Warning: Ion {ion} not in HDF5, skipping")
            continue

        # -------------------------
        # Ion XY column density map
        # -------------------------

        plotter.plot_xy(
            column_density_values=cd_data.ions[ion]["column_density"],
            ion=ion,
            log_scale=True
        )


        ax = plotter.plot_cddf_hist(
            cddf=cd_data.ions[ion]["cddf"],
            bin_centers=cd_data.ions[ion]["bin_centers"],
            bin_width=cd_data.ions[ion]["bin_width"],
            ion=ion,
            log_scale=True
        )
        # --- CDDF single plotting inside the original data directory ---
        ax = plot_eagle_cddf(ax, ion, cfg_plot)

        

        plt.tight_layout()

        file_path = output_dir / f"CDDF/{ion}.png"
        file_path = output_dir / f"CDDF/{ion}.png"

        file_path.parent.mkdir(parents=True, exist_ok=True)


        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")

        plt.close(ax.figure)

        print("Saved", file_path)

    print(f"Finished replotting single HDF5: {data_file}")


def run_multiple(cfg_plot):
    """
    Replot multiple HDF5 files on the same figure for one element and its ions.

    Parameters
    ----------
    cfg_plot : PlotConfig
        A single PlotConfig object with:
            - label_criterion: str (used for legend)
            - hdf5_files: list of paths to HDF5 files
            - output_directory: where to save combined plots
    """
    element_ax_dict = {}
    ion_ax_dict = {}

    for data_file in cfg_plot.data_files:
        data_file = Path(data_file)

        # --- Load HDF5 data ---
        cd_data = single_cd(data_file)
        data_unpacker = unwrapper(cd_data.cfg)

        # --- Initialize plotter ---
        plotter = plot.column_density_plotter(
            cfg=cd_data.cfg,
            data_unpacker=data_unpacker,
            x_edges=cd_data.xedges,
            y_edges=cd_data.yedges,
            cfg_plot=cfg_plot
        )
        # this is the label for this particular data_file, based on the selection criterion
        label = get_label(cd_data, data_unpacker, cfg_plot.label_criterion)
        # --- Element CDDF ---
        ax_elem = element_ax_dict.get(cd_data.element_name)
        ax_elem = plotter.plot_cddf_hist(
            cddf=cd_data.element_cddf,
            bin_centers=cd_data.element_bin_centers,
            bin_width=cd_data.element_bin_width,
            element=cd_data.element_name,
            label=label,
            log_scale=True,
            ax=ax_elem
        )
        element_ax_dict[cd_data.element_name] = ax_elem

        # --- Ion CDDFs ---
        ions_to_plot = cd_data.cfg.chemistry.ion
        for ion in ions_to_plot:
            #ion is not present in the HDF5 file, so it is somewhere we do not know
            if ion not in cd_data.ions:
                print(f"Warning: Ion {ion} not in HDF5, skipping")
                continue
            ax_ion = ion_ax_dict.get(ion)
            ax_ion = plotter.plot_cddf_hist(
                cddf=cd_data.ions[ion]["cddf"],
                bin_centers=cd_data.ions[ion]["bin_centers"],
                bin_width=cd_data.ions[ion]["bin_width"],
                ion=ion,
                label=label,
                log_scale=True,
                ax=ax_ion
            )
            
            ion_ax_dict[ion] = ax_ion

    # --- Save combined element CDDF plot ---

    output_dir = Path(cfg_plot.data_directory) /cfg_plot.output_directory / f"CDDF_{cfg_plot.label_criterion}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for element_name, ax in element_ax_dict.items():
        ax.legend()
        ax.set_title(f"CDDF of {element_name}")
        plt.tight_layout()
        file_path = output_dir / f"{element_name}.png"
        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved combined element CDDF: {file_path}")

    # --- Save combined ion CDDF plots ---
    for ion_name, ax in ion_ax_dict.items():
        # Overlay EAGLE data if requested
        ax = plot_eagle_cddf(ax, ion_name, cfg_plot)
        ax.legend()
        ax.set_title(f"CDDF of {ion_name}")
        plt.tight_layout()
        file_path = output_dir / f"{ion_name}.png"
        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved combined ion CDDF: {file_path}")