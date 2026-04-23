# spec_analysis/column_density/replot.py

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
from spec_analysis.unpack_data import single_cd, unwrapper
from spec_analysis import plot  # your column_density_plotter class
from spec_analysis.data_structure.plot import plot_config
import numpy as np
import pandas as pd


def get_slice_thickness_cMpc(cd_data):
    """Return projection slice thickness in cMpc, or None if unavailable."""
    if hasattr(cd_data.cfg.window, "projection_slices"):
        slices = cd_data.cfg.window.projection_slices
        if slices:
            return cd_data.cfg.simulation.box_length / slices
    return None


def get_label(cd_data, data_unpacker, selection_criterium):

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
    elif selection_criterium == "compare_slices":
        return cd_data.simulation_name
    #can add more criteria here as needed
    else:
        raise ValueError(f"Unknown label criterion: {selection_criterium}")


def get_label_and_style(cd_data, data_unpacker, cfg_plot):
    """Return label, linestyle, and slice thickness based on selection criterion and filename stem rules."""
    label = get_label(cd_data, data_unpacker, cfg_plot.label_criterion)
    line_style = "-"
    slice_thickness = get_slice_thickness_cMpc(cd_data)

    if getattr(cfg_plot, "stack_total_label", False):
        stem = Path(cd_data.hdf5_path).stem
        if not label.endswith(stem):
            if stem == "total":
                name = "full box projection"
            if stem == "stacked": 
                name = f"{slice_thickness:.2f} cMpc" if slice_thickness else "slice average"
            label = f"{label} [{name}]"
        if stem == "total":
            line_style = "--"
            
    elif getattr(cfg_plot, "Z_label", False):
        if getattr(cd_data.cfg.chemistry, "metallicity", True):
            name = "" #Maybe another tag but for now empty
        else:
            name = rf"[0.1 $Z_\odot$]"
            line_style = "--"
        label = f"{label} {name}"
        
    elif getattr(cfg_plot, "slice_label", False) and not getattr(cfg_plot, "stack_total_label", False):
        if hasattr(cd_data.cfg.window, "projection_slices"):
            name = f"{slice_thickness:.2f} cMpc" if slice_thickness else "slice average"
            label = f"{label} [{name}]"

    return label, line_style, slice_thickness


def load_eagle_cddf(ion, cfg_plot, eagle_directory=None):
    """Return EAGLE CDDF points as (logN, log10f) or None if unavailable."""
    if not getattr(cfg_plot, "plot_eagle", False):
        return None

    if eagle_directory is None:
        eagle_directory = getattr(cfg_plot, "eagle_cddf_directory", None)

    if eagle_directory is None:
        return None

    file_path = Path(eagle_directory) / f"{ion}.csv"
    if not file_path.exists():
        print(f"EAGLE file not found for {ion}, skipping.")
        return None

    data = np.loadtxt(file_path, delimiter=",")
    if data.ndim == 1:
        data = np.atleast_2d(data)
    if data.shape[1] < 2:
        print(f"EAGLE file for {ion} has unexpected format, skipping.")
        return None

    return data[:, 0], data[:, 1]


def load_observational_cddf(ion, cfg_plot):
    """Return observational CDDF CSV data for an ion as a list of (label, dataframe)."""
    if not getattr(cfg_plot, "plot_observations", False):
        return None

    if getattr(cfg_plot, "observational_cddf_directory", None) is None:
        return None

    ion_directory = Path(cfg_plot.observational_cddf_directory) / ion
    if not ion_directory.exists() or not ion_directory.is_dir():
        print(f"Observational directory not found for {ion}, skipping.")
        return None

    csv_files = sorted(ion_directory.glob("*.csv"))
    if not csv_files:
        print(f"No observational CSV files found for {ion}, skipping.")
        return None

    observational_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"Failed to read observational file {file_path}: {exc}")
            continue

        if "logN" not in df.columns or "logf" not in df.columns:
            print(f"Observational file {file_path} is missing logN/logf columns, skipping.")
            continue

        observational_data.append((file_path.stem, df))

    return observational_data or None


def plot_eagle_cddf(ax, ion, cfg_plot):
    """
    Overlay EAGLE CDDF data if cfg_plot.plot_eagle is True
    and the corresponding CSV file exists.

    """
    if getattr(cfg_plot, "Z_label", False):
        base_dir = getattr(cfg_plot, "eagle_cddf_directory", None)
        if base_dir is None:
            return ax

        eagle_data = load_eagle_cddf(ion, cfg_plot, eagle_directory=base_dir)
        if eagle_data is not None:
            eagle_x, eagle_y = eagle_data
            ax.plot(
                eagle_x,
                eagle_y,
                label="EAGLE",
                color="black",

            )

        no_z_dir = Path(base_dir) / "no_Z"
        eagle_no_z_data = load_eagle_cddf(ion, cfg_plot, eagle_directory=no_z_dir)
        if eagle_no_z_data is not None:
            eagle_x, eagle_y = eagle_no_z_data
            ax.plot(
                eagle_x,
                eagle_y,
                label=rf"EAGLE [0.1 $Z_\odot$]",
                linestyle="--",
                color="black",

            )

        return ax

    else:
        eagle_data = load_eagle_cddf(ion, cfg_plot)
        if eagle_data is None:
            return ax

        eagle_x, eagle_y = eagle_data

        if cfg_plot.label_criterion == "compare_slices":
            label="EAGLE (L100/m6 [6.25 cMpc slice average])"
        else:
            label="EAGLE"

        ax.plot(
            eagle_x,
            eagle_y,
            label=label,
            color="black",

        )

        return ax
    

def plot_observational_cddf(
    ax,
    ion,
    cfg_plot,
    observational_data: Optional[List[Tuple[str, pd.DataFrame]]] = None,):
    """Plot observational CDDF CSV files for an ion."""
    if getattr(cfg_plot, "label_criterion", None) == "compare_slices":
        return ax

    if observational_data is None:
        observational_data = load_observational_cddf(ion, cfg_plot)

    if not observational_data:
        return ax

    for label, df in observational_data:
        logN = df["logN"].values
        logf = df["logf"].values

        xerr = df["xerr"].values if "xerr" in df.columns else None
        if "yerr_plus" in df.columns and "yerr_minus" in df.columns:
            yerr = [df["yerr_minus"].values, df["yerr_plus"].values]
        elif "yerr" in df.columns:
            yerr = df["yerr"].values
        else:
            yerr = None

        ax.errorbar(
            logN,
            logf,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            capsize=3,
            label=label,
        )

    return ax


def run_single_halo(cfg_plot: plot_config):
    """Replot a single galaxy/halo with transverse radial profiles."""
    data_file = Path(cfg_plot.data_files[0])

    cd_data = single_cd(data_file, load_cd=cfg_plot.load_cd)
    data_unpacker = unwrapper(cd_data.cfg)

    if cd_data.halo is None:
        raise ValueError("Expected halo metadata in HDF5 file, but none found.")

    

    # Plot transverse profiles for element
    plotter = plot.column_density_plotter(
        cfg=cd_data.cfg,
        data_unpacker=data_unpacker,
        x_edges=cd_data.xedges,
        y_edges=cd_data.yedges,
        cfg_plot=cfg_plot,
    )

    output_dir = Path(data_unpacker.output_directory)
    trans_dir = output_dir / "transverse_profiles"/f"halo_{int(np.asarray(cd_data.halo['catalogue_id']).item())}"
    trans_dir.mkdir(parents=True, exist_ok=True)

    if cd_data.element_name in cd_data.transverse_profiles:
        profile = cd_data.transverse_profiles[cd_data.element_name]
        ax = plotter.plot_radial_transverse(
            r_centers=profile["r_centers"],
            column_density=profile["column_density"],
            element=cd_data.element_name,
        )
        ax.figure.tight_layout()
        file_path = trans_dir / f"{cd_data.element_name}_radial.png"
        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print("Saved", file_path)

    # Plot transverse profiles for ions
    for ion in cd_data.ions.keys():
        if ion in cd_data.transverse_profiles:
            profile = cd_data.transverse_profiles[ion]
            ax = plotter.plot_radial_transverse(
                r_centers=profile["r_centers"],
                column_density=profile["column_density"],
                ion=ion,
            )
            ax.figure.tight_layout()
            file_path = trans_dir / f"{ion}_radial.png"
            ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            print("Saved", file_path)

    print(f"Halo {cd_data.halo['catalogue_id']} plot complete.")

    # --- 2D Column Density Histograms ---
    if cfg_plot.plot_2d_histogram and cfg_plot.load_cd:
        halo_id = int(np.asarray(cd_data.halo['catalogue_id']).item())
        cd2d_dir = output_dir / "2d_column_density" / f"halo_{halo_id}"
        cd2d_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(cd_data, 'element_cd') and cd_data.element_cd is not None:
            ax = plotter.plot_xy(
                column_density_values=cd_data.element_cd,
                element=cd_data.element_name,
                log_scale=True,
                ax=None
            )
            ax.figure.tight_layout()
            file_path = cd2d_dir / f"{cd_data.element_name}_2d.png"
            ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            print("Saved", file_path)

        for ion in cd_data.ions.keys():
            ion_cd = cd_data.ions[ion].get("column_density") if isinstance(cd_data.ions[ion], dict) else None
            if ion_cd is not None:
                ax = plotter.plot_xy(
                    column_density_values=ion_cd,
                    ion=ion,
                    log_scale=True,
                    ax=None
                )
                ax.figure.tight_layout()
                file_path = cd2d_dir / f"{ion}_2d.png"
                ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close(ax.figure)
                print("Saved", file_path)


def run_multiple_halos(cfg_plot: plot_config):
    """Replot multiple galaxies/halos with transverse radial profiles."""
    for data_file in cfg_plot.data_files:
        cd_data = single_cd(data_file, load_cd=cfg_plot.load_cd)
        data_unpacker = unwrapper(cd_data.cfg)

        if cd_data.halo is None:
            raise ValueError("Expected halo metadata in HDF5 file, but none found.")

        # Plot transverse profiles for element
        plotter = plot.column_density_plotter(
            cfg=cd_data.cfg,
            data_unpacker=data_unpacker,
            x_edges=cd_data.xedges,
            y_edges=cd_data.yedges,
            cfg_plot=cfg_plot,
        )

        output_dir = Path(data_unpacker.output_directory)
        trans_dir = output_dir / "transverse_profiles"/f"halo_{int(np.asarray(cd_data.halo['catalogue_id']).item())}"
        trans_dir.mkdir(parents=True, exist_ok=True)

        if cd_data.element_name in cd_data.transverse_profiles:
            profile = cd_data.transverse_profiles[cd_data.element_name]
            ax = plotter.plot_radial_transverse(
                r_centers=profile["r_centers"],
                column_density=profile["column_density"],
                element=cd_data.element_name,
            )
            ax.figure.tight_layout()
            file_path = trans_dir / f"{cd_data.element_name}_radial.png"
            ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            print("Saved", file_path)

        # Plot transverse profiles for ions
        for ion in cd_data.ions.keys():
            if ion in cd_data.transverse_profiles:
                profile = cd_data.transverse_profiles[ion]
                ax = plotter.plot_radial_transverse(
                    r_centers=profile["r_centers"],
                    column_density=profile["column_density"],
                    ion=ion,
                )
                ax.figure.tight_layout()
                file_path = trans_dir / f"{ion}_radial.png"
                ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close(ax.figure)
                print("Saved", file_path)

        print(f"Halo {cd_data.halo['catalogue_id']} plot complete.")

        # --- 2D Column Density Histograms ---
        if cfg_plot.plot_2d_histogram and cfg_plot.load_cd:
            halo_id = int(np.asarray(cd_data.halo['catalogue_id']).item())
            cd2d_dir = output_dir / "2d_column_density" / f"halo_{halo_id}"
            cd2d_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(cd_data, 'element_cd') and cd_data.element_cd is not None:
                ax = plotter.plot_xy(
                    column_density_values=cd_data.element_cd,
                    element=cd_data.element_name,
                    log_scale=True,
                    ax=None
                )
                ax.figure.tight_layout()
                file_path = cd2d_dir / f"{cd_data.element_name}_2d.png"
                ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close(ax.figure)
                print("Saved", file_path)
            else:
                print(f"No column density map found for element {cd_data.element_name} in halo {halo_id}, skipping 2D plot.")

            for ion in cd_data.ions.keys():
                ion_cd = cd_data.ions[ion].get("column_density") if isinstance(cd_data.ions[ion], dict) else None
                if ion_cd is not None:
                    ax = plotter.plot_xy(
                        column_density_values=ion_cd,
                        ion=ion,
                        log_scale=True,
                        ax=None
                    )
                    ax.figure.tight_layout()
                    file_path = cd2d_dir / f"{ion}_2d.png"
                    ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
                    plt.close(ax.figure)
                    print("Saved", file_path)
                else:
                    print(f"No column density map found for ion {ion} in halo {halo_id}, skipping 2D plot.")
    print("All halos plotted.")

def run_single(cfg_plot: plot_config):

    data_file = Path(cfg_plot.data_files[0])
    compare_slices_mode = getattr(cfg_plot, "label_criterion", None) == "compare_slices"

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
    label, line_style, current_slice_thickness = get_label_and_style(cd_data, data_unpacker, cfg_plot)
    
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
        cddf=cd_data.element_cddf,
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
            cddf=cd_data.ions[ion]["cddf"],
            bin_centers=cd_data.ions[ion]["bin_centers"],
            bin_width=cd_data.ions[ion]["bin_width"],
            ion=ion,
            label=label,
            linestyle=line_style,
            log_scale=True
        )
        # --- CDDF single plotting inside the original data directory ---
        if not compare_slices_mode:
            ax = plot_eagle_cddf(ax, ion, cfg_plot)
            ax = plot_observational_cddf(ax, ion, cfg_plot)

        plt.legend()
        ax.set_title(f"CDDF of {ion}")

        plt.tight_layout()

        file_path = output_dir / f"CDDF/{ion}.png"
        file_path = output_dir / f"CDDF/{ion}.png"

        file_path.parent.mkdir(parents=True, exist_ok=True)


        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")

        plt.close(ax.figure)

        print("Saved", file_path)

        if getattr(cfg_plot, "plot_eagle", False) and not compare_slices_mode:
            eagle_data = load_eagle_cddf(ion, cfg_plot)
            if eagle_data is not None:
                eagle_x, eagle_y = eagle_data
                fig_diff, ax_diff = plt.subplots(figsize=(7, 6))
                ax_diff = plotter.plot_cddf_difference(
                    ax=ax_diff,
                    sim_bin_centers=cd_data.ions[ion]["bin_centers"],
                    sim_cddf=cd_data.ions[ion]["cddf"],
                    eagle_x=eagle_x,
                    eagle_log_cddf=eagle_y,
                    label=label,
                    linestyle=line_style,
                )
                ax_diff.legend()
                ax_diff.set_title(f"CDDF difference vs EAGLE of {ion}")
                fig_diff.tight_layout()
                diff_path = output_dir /f"{ion}_delta_vs_eagle.png"
                diff_path.parent.mkdir(parents=True, exist_ok=True)
                fig_diff.savefig(diff_path, dpi=300, bbox_inches="tight")
                plt.close(fig_diff)
                print("Saved", diff_path)

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
    ion_diff_ax_dict = {}
    eagle_data_cache = {}
    observational_data_cache = {}
    simulation_color_map = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    compare_slices_mode = getattr(cfg_plot, "label_criterion", None) == "compare_slices"
    baseline = None
    baseline_slice_thickness = None
    baseline_file = None


    def get_simulation_color(color_key):
        if color_key not in simulation_color_map:
            simulation_color_map[color_key] = color_cycle[
                len(simulation_color_map) % len(color_cycle)
            ]
        return simulation_color_map[color_key]

    # Load baseline for compare_slices mode
    if compare_slices_mode and cfg_plot.data_files:
        baseline_file = Path(cfg_plot.data_files[0])
        baseline_cd = single_cd(baseline_file, load_cd=cfg_plot.load_cd)
        baseline = {
            "element_name": baseline_cd.element_name,
            "element_bin_centers": baseline_cd.element_bin_centers,
            "element_cddf": baseline_cd.element_cddf,
            "ions": {},
        }
        for ion_name, ion_payload in baseline_cd.ions.items():
            if isinstance(ion_payload, dict) and "bin_centers" in ion_payload and "cddf" in ion_payload:
                baseline["ions"][ion_name] = {
                    "bin_centers": ion_payload["bin_centers"],
                    "cddf": ion_payload["cddf"],
                }
        baseline_slice_thickness = get_slice_thickness_cMpc(baseline_cd)



    for data_file in cfg_plot.data_files:
        data_file = Path(data_file)
        is_baseline_file = compare_slices_mode and baseline_file is not None and data_file == baseline_file

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
        label, line_style, current_slice_thickness = get_label_and_style(cd_data, data_unpacker, cfg_plot)
        plot_label = None if is_baseline_file else label
        """
        if getattr(cfg_plot, "slice_label", False) and hasattr(cd_data.cfg.window, "projection_slices"):
            color_key = (cd_data.simulation_name, cd_data.cfg.window.projection_slices)

        """
        if getattr(cfg_plot, "Z_label", False):
            
            color_key = cd_data.simulation_name
        else:
            color_key = np.random.choice(np.linspace(0, 1, num=1000))  # fallback to random color if no slice info
        
        line_color = get_simulation_color(color_key)
        # --- Element CDDF ---
        ax_elem = element_ax_dict.get(cd_data.element_name)
        if compare_slices_mode and baseline is not None and cd_data.element_name == baseline["element_name"]:
            if ax_elem is None:
                _, ax_elem = plt.subplots(figsize=(7, 6))
            ax_elem = plotter.plot_cddf_difference(
                ax=ax_elem,
                sim_bin_centers=cd_data.element_bin_centers,
                sim_cddf=cd_data.element_cddf,
                eagle_x=np.array([]),
                eagle_log_cddf=np.array([]),
                label=plot_label,
                linestyle=line_style,
                color=line_color,
                baseline_cddf=baseline["element_cddf"],
                baseline_bin_centers=baseline["element_bin_centers"],
            )
        else:
            ax_elem = plotter.plot_cddf_hist(
                cddf=cd_data.element_cddf,
                bin_centers=cd_data.element_bin_centers,
                bin_width=cd_data.element_bin_width,
                element=cd_data.element_name,
                label=plot_label,
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
            if compare_slices_mode and baseline is not None and ion in baseline["ions"]:
                if ax_ion is None:
                    _, ax_ion = plt.subplots(figsize=(7, 6))

                ax_ion = plotter.plot_cddf_difference(
                    ax=ax_ion,
                    sim_bin_centers=cd_data.ions[ion]["bin_centers"],
                    sim_cddf=cd_data.ions[ion]["cddf"],
                    eagle_x=np.array([]),
                    eagle_log_cddf=np.array([]),
                    label=plot_label,
                    linestyle=line_style,
                    color=line_color,
                    baseline_cddf=baseline["ions"][ion]["cddf"],
                    baseline_bin_centers=baseline["ions"][ion]["bin_centers"],
                )
            else:
                ax_ion = plotter.plot_cddf_hist(
                    cddf=cd_data.ions[ion]["cddf"],
                    bin_centers=cd_data.ions[ion]["bin_centers"],
                    bin_width=cd_data.ions[ion]["bin_width"],
                    ion=ion,
                    label=plot_label,
                    linestyle=line_style,
                    color=line_color,
                    log_scale=True,
                    ax=ax_ion
                )
            
            ion_ax_dict[ion] = ax_ion

            if getattr(cfg_plot, "plot_eagle", False) and not compare_slices_mode:
                if getattr(cfg_plot, "Z_label", False):
                    base_dir = getattr(cfg_plot, "eagle_cddf_directory", None)
                    if base_dir is None:
                        eagle_data = None
                    # compare to either constant metallicity or metallicity-dependent EAGLE CDDF based on the HDF5 filename and config
                    else:
                        if "no_Z" in Path(cd_data.hdf5_path).parts or not getattr(cd_data.cfg.chemistry, "metallicity", True):
                            eagle_directory = Path(base_dir) / "no_Z"
                        else:
                            eagle_directory = base_dir

                        cache_key = (ion, str(eagle_directory))
                        if cache_key not in eagle_data_cache:
                            eagle_data_cache[cache_key] = load_eagle_cddf(
                                ion,
                                cfg_plot,
                                eagle_directory=eagle_directory,
                            )
                        eagle_data = eagle_data_cache[cache_key]
                else:
                    if ion not in eagle_data_cache:
                        eagle_data_cache[ion] = load_eagle_cddf(ion, cfg_plot)
                    eagle_data = eagle_data_cache[ion]

                if eagle_data is not None:
                    eagle_x, eagle_y = eagle_data
                    ax_diff = ion_diff_ax_dict.get(ion)
                    if ax_diff is None:
                        _, ax_diff = plt.subplots(figsize=(7, 6))
                    ax_diff = plotter.plot_cddf_difference(
                        ax=ax_diff,
                        sim_bin_centers=cd_data.ions[ion]["bin_centers"],
                        sim_cddf=cd_data.ions[ion]["cddf"],
                        eagle_x=eagle_x,
                        eagle_log_cddf=eagle_y,
                        label=label,
                        linestyle=line_style,
                        color=line_color,
                    )
                    ion_diff_ax_dict[ion] = ax_diff


    # --- Save combined element CDDF plot ---

    output_dir = Path(cfg_plot.data_directory) /cfg_plot.output_directory / f"CDDF_{cfg_plot.label_criterion}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for element_name, ax in element_ax_dict.items():
        ax.legend()
        if compare_slices_mode:
            base_txt = f"{baseline_slice_thickness:.2f} cMpc" if baseline_slice_thickness is not None else "baseline"
            ax.set_title(f"CDDF of {element_name} relative to baseline slice ({base_txt})")
        else:
            ax.set_title(f"CDDF of {element_name}")
        plt.tight_layout()
        file_path = output_dir / f"{element_name}.png"
        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved combined element CDDF: {file_path}")

    # --- Save combined ion CDDF plots ---
    for ion_name, ax in ion_ax_dict.items():
        # Overlay EAGLE data if requested
        if not compare_slices_mode:
            ax = plot_eagle_cddf(ax, ion_name, cfg_plot)
            if getattr(cfg_plot, "plot_observations", False):
                if ion_name not in observational_data_cache:
                    observational_data_cache[ion_name] = load_observational_cddf(ion_name, cfg_plot)
                ax = plot_observational_cddf(
                    ax,
                    ion_name,
                    cfg_plot,
                    observational_data=observational_data_cache[ion_name],
                )
        
        ax.legend()
        if compare_slices_mode:
            base_txt = f"{baseline_slice_thickness:.2f} cMpc" if baseline_slice_thickness is not None else "baseline"
            ax.set_title(f"CDDF of {ion_name} relative to baseline slice ({base_txt})")
        else:
            ax.set_title(f"CDDF of {ion_name}")
        plt.tight_layout()
        file_path = output_dir / f"{ion_name}.png"
        ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved combined ion CDDF: {file_path}")

    if getattr(cfg_plot, "plot_eagle", False):
        for ion_name, ax in ion_diff_ax_dict.items():
            ax.legend()
            ax.set_title(f"CDDF difference vs EAGLE of {ion_name}")
            plt.tight_layout()
            file_path = output_dir /f"{ion_name}_delta_vs_eagle.png"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            print(f"Saved combined ion CDDF difference: {file_path}")