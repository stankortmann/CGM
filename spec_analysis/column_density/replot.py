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
    elif selection_criterium == "file_name":
        return f"Filename: {Path(cd_data.hdf5_path).stem}"
    elif selection_criterium == "scale_factor":
        return f"a= {data_unpacker.scale_factor:.3f}"
    elif selection_criterium == "particle_resolution":
        return f"m{cd_data.cfg.simulation.resolution}"
    elif selection_criterium == "pixel_resolution":
        pixel_size_ckpc = cd_data.cfg.simulation.box_length * 1000 / cd_data.cfg.window.resolution
        return rf"Pixel size: {pixel_size_ckpc:.1f}$^2 cKpc^2$"
    elif selection_criterium == "simulation_name":
        return cd_data.simulation_name
    #can add more criteria here as needed
    else:
        raise ValueError(f"Unknown label criterion: {selection_criterium}")


def get_label_and_style(cd_data, data_unpacker, cfg_plot):
    """Return label and linestyle based on selection criterion and filename stem rules."""
    label = get_label(cd_data, data_unpacker, cfg_plot.label_criterion)
    line_style = "-"
    

    if getattr(cfg_plot, "stack_total_label", False) and not getattr(cfg_plot, "slice_label", False):
        stem = Path(cd_data.hdf5_path).stem
        if not label.endswith(stem):
            if stem == "total":
                name = "full box projection"
            if stem == "stacked": 
                name = f"{cd_data.cfg.simulation.box_length/cd_data.cfg.window.projection_slices:.2f} cMpc slice average"
            label = f"{label} [{name}]"
        if stem == "stacked":
            line_style = "--"
            
    if getattr(cfg_plot, "Z_label", False):

        if getattr(cd_data.cfg.chemistry, "metallicity", True):
            name = "" #Maybe another tag but for now empty
        else:
            name = rf"[0.1 $Z_\odot$]"
            line_style = "--"
        
        label = f"{label} {name}"
        
    if getattr(cfg_plot, "slice_label", False):
        if hasattr(cd_data.cfg.window, "projection_slices"):
            name = f"{cd_data.cfg.simulation.box_length/cd_data.cfg.window.projection_slices:.2f} cMpc slice average"
            label = f"{label} [{name}]"

    return label, line_style


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
        label="EAGLE (L100/m6 [6.25 cMpc slice average])",
        marker="x",
        s=20,
        color="black",
        zorder=5
    )

    return ax




def run_single_halo(cfg_plot: plot_config):
    """Replot a single galaxy/halo with transverse radial profiles."""
    data_file = Path(cfg_plot.data_files[0])

    cd_data = single_cd(data_file, load_cd=cfg_plot.load_cd)
    data_unpacker = unwrapper(cd_data.cfg)

    if cd_data.halo is None:
        raise ValueError("Expected halo metadata in HDF5 file, but none found.")

    output_dir = Path(data_unpacker.output_directory) / "replot"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot transverse profiles for element
    plotter = plot.column_density_plotter(
        cfg=cd_data.cfg,
        data_unpacker=data_unpacker,
        x_edges=cd_data.xedges,
        y_edges=cd_data.yedges,
        cfg_plot=cfg_plot,
    )

    if cd_data.element_name in cd_data.transverse_profiles:
        profile = cd_data.transverse_profiles[cd_data.element_name]
        plotter.plot_radial_transverse(
            r_centers=profile["r_centers"],
            column_density=profile["column_density"],
            element=cd_data.element_name,
        )

    # Plot transverse profiles for ions
    for ion in cd_data.ions.keys():
        if ion in cd_data.transverse_profiles:
            profile = cd_data.transverse_profiles[ion]
            plotter.plot_radial_transverse(
                r_centers=profile["r_centers"],
                column_density=profile["column_density"],
                ion=ion,
            )

    print(f"Halo {cd_data.halo['catalogue_id']} replot complete.")


def run_single(cfg_plot: plot_config):

    data_file = Path(cfg_plot.data_files[0])

    cd_data = single_cd(data_file, load_cd=cfg_plot.load_cd)
    data_unpacker = unwrapper(cd_data.cfg)

    plotter = plot.column_density_plotter(
        cfg=cd_data.cfg,
        data_unpacker=data_unpacker,
        x_edges=cd_data.xedges,
        y_edges=cd_data.yedges,
        cfg_plot=cfg_plot
    )
    # Get label for this plot based on the selection criterion
    label, line_style = get_label_and_style(cd_data, data_unpacker, cfg_plot)
    
    # -------------------------
    # Element XY column density map
    # -------------------------
    """ 
    plotter.plot_xy(
        column_density_values=cd_data.element_cd,
        element=cd_data.element_name,
        log_scale=True
    )
    """ 
    output_dir = Path(data_unpacker.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # -------------------------
    # Element CDDF
    # -------------------------

    ax = plotter.plot_cddf_hist(
        cddf=cd_data.element_cddf*16,
        bin_centers=cd_data.element_bin_centers,
        bin_width=cd_data.element_bin_width,
        element=cd_data.element_name,
        label=label,
        linestyle=line_style,
        log_scale=True
    )
    plt.legend()
    ax.set_title(f"CDDF of {cd_data.element_name}")
    plt.tight_layout()
    file_path = output_dir / f"CDDF/{cd_data.element_name}.png"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    print("Saved", file_path)


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
        """
        plotter.plot_xy(
            column_density_values=cd_data.ions[ion]["column_density"],
            ion=ion,
            log_scale=True
        )
        """

        ax = plotter.plot_cddf_hist(
            cddf=cd_data.ions[ion]["cddf"]*16,
            bin_centers=cd_data.ions[ion]["bin_centers"],
            bin_width=cd_data.ions[ion]["bin_width"],
            ion=ion,
            label=label,
            linestyle=line_style,
            log_scale=True
        )
        # --- CDDF single plotting inside the original data directory ---
        ax = plot_eagle_cddf(ax, ion, cfg_plot)

        plt.legend()
        ax.set_title(f"CDDF of {ion}")

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
    simulation_color_map = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])


    def get_simulation_color(color_key):
        if color_key not in simulation_color_map:
            simulation_color_map[color_key] = color_cycle[
                len(simulation_color_map) % len(color_cycle)
            ]
        return simulation_color_map[color_key]



    for data_file in cfg_plot.data_files:
        data_file = Path(data_file)

        # --- Load HDF5 data ---
        cd_data = single_cd(data_file, load_cd=cfg_plot.load_cd)
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
        label, line_style = get_label_and_style(cd_data, data_unpacker, cfg_plot)

        if getattr(cfg_plot, "slice_label", False) and hasattr(cd_data.cfg.window, "projection_slices"):
            color_key = (cd_data.simulation_name, cd_data.cfg.window.projection_slices)
        else:
            color_key = cd_data.simulation_name

        line_color = get_simulation_color(color_key)
        # --- Element CDDF ---
        ax_elem = element_ax_dict.get(cd_data.element_name)
        ax_elem = plotter.plot_cddf_hist(
            cddf=cd_data.element_cddf,
            bin_centers=cd_data.element_bin_centers,
            bin_width=cd_data.element_bin_width,
            element=cd_data.element_name,
            label=label,
            linestyle=line_style,
            color=line_color,
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
                linestyle=line_style,
                color=line_color,
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