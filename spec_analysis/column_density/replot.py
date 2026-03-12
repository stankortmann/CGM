# spec_analysis/column_density/replot.py

import matplotlib.pyplot as plt
from pathlib import Path
from spec_analysis.unpack_data import single_cd, unwrapper
from spec_analysis import plot  # your column_density_plotter class
from spec_analysis.data_structure_plot import plot_config


def get_label(cd_data,data_unpacker,selection_criterium):

    if selection_criterium == "box_size":
        return f"Box size: {cd_data.cfg.simulation.box_size:.1f} Mpc"
    elif selection_criterium == "redshift":
        return f"z= {data_unpacker.redshift:.2f}"
    elif selection_criterium == "scale_factor":
        return f"a= {data_unpacker.scale_factor:.3f}"
    elif selection_criterium == "resolution_particles":
        return f"m: {cd_data.cfg.simulation.resolution}"
    elif selection_criterium == "resolution_pixels":
        return f"Pixel number: {cd_data.cfg.simulation.resolution**2}"
    elif selection_criterium == "simulation_name":
        return f"L{cd_data.cfg.simulation.box_length:03d}_m{cd_data.cfg.simulation.resolution}_{cd_data.cfg.simulation.name}"

    else:
        raise ValueError(f"Unknown label criterion: {selection_criterium}")






def run_single(plot_cfg: plot_config):
    """
    Replot a single HDF5 file using the original column_density_plotter class.
    """
    hdf5_file = Path(plot_cfg.hdf5_files[0])  # Assuming only one file for single plot
    cd_data = single_cd(hdf5_file)
    data_unpacker = unwrapper(cd_data.cfg)

    plotter = plot.column_density_plotter(
        cfg=cd_data.cfg,
        data_unpacker=data_unpacker,
        x_edges=cd_data.xedges,
        y_edges=cd_data.yedges
    )

    # --- Element ---
    plotter.plot_xy(
        column_density_values=cd_data.element_cd.value,
        element=cd_data.element_name,
        log_scale=True
    )
    plotter.plot_cddf_hist(
        cddf=cd_data.element_cddf,
        bin_centers=cd_data.element_bin_centers,
        bin_width=cd_data.element_bin_width,
        element=cd_data.element_name,
        log_scale=True
    )

    # --- Ions ---
    ions_to_plot = cd_data.cfg["chemistry"].get("ion", [])
    for ion in ions_to_plot:
        if ion not in cd_data.ions:
            print(f"Warning: Ion {ion} not in HDF5, skipping")
            continue
        plotter.plot_xy(
            column_density_values=cd_data.ions[ion]["column_density"].value,
            ion=ion,
            log_scale=True
        )
        plotter.plot_cddf_hist(
            cddf=cd_data.ions[ion]["cddf"],
            bin_centers=cd_data.ions[ion]["bin_centers"],
            bin_width=cd_data.ions[ion]["bin_width"],
            ion=ion,
            log_scale=True
        )

    print(f"Finished replotting single HDF5: {hdf5_file}")


def run_multiple(plot_cfg):
    """
    Replot multiple HDF5 files on the same figure for one element and its ions.

    Parameters
    ----------
    plot_cfg : PlotConfig
        A single PlotConfig object with:
            - label_criterion: str (used for legend)
            - hdf5_files: list of paths to HDF5 files
            - output_directory: where to save combined plots
    """
    element_ax_dict = {}
    ion_ax_dict = {}

    for hdf5_file in plot_cfg.hdf5_files:
        hdf5_file = Path(hdf5_file)

        # --- Load HDF5 data ---
        cd_data = single_cd(hdf5_file)
        data_unpacker = unwrapper(cd_data.cfg)

        # --- Initialize plotter ---
        plotter = plot.column_density_plotter(
            cfg=cd_data.cfg,
            data_unpacker=data_unpacker,
            x_edges=cd_data.xedges,
            y_edges=cd_data.yedges
        )

        # --- Element CDDF ---
        ax_elem = element_ax_dict.get(cd_data.element_name)
        ax_elem = plotter.plot_cddf_hist(
            cddf=cd_data.element_cddf,
            bin_centers=cd_data.element_bin_centers,
            bin_width=cd_data.element_bin_width,
            element=cd_data.element_name,
            label=plot_cfg.label_criterion,
            log_scale=True,
            ax=ax_elem,
            save=False
        )
        element_ax_dict[cd_data.element_name] = ax_elem

        # --- Ion CDDFs ---
        ions_to_plot = cd_data.cfg["chemistry"].get("ion", [])
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
                label=plot_cfg.label_criterion,
                log_scale=True,
                ax=ax_ion,
                save=False
            )
            ion_ax_dict[ion] = ax_ion

    # --- Save combined element CDDF plot ---

    output_dir = Path(plot_cfg.data_directory) /plot_cfg.output_directory / f"CDDF_{cfg_plot.label_criterion}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for element_name, ax in element_ax_dict.items():
        plt.tight_layout()
        file_path = output_dir / f"{element_name}.png"
        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved combined element CDDF: {file_path}")

    # --- Save combined ion CDDF plots ---
    for ion_name, ax in ion_ax_dict.items():
        plt.tight_layout()
        file_path = output_dir / f"{ion_name}.png"
        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved combined ion CDDF: {file_path}")