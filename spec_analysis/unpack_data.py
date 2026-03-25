import pandas
import unyt as u
import h5py
import gc
import numpy as np
from pathlib import Path
from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as swift


from pathlib import Path
import pandas
import swiftsimio as swift
from swiftsimio import load
import json
from spec_analysis import plot
from swiftsimio.objects import cosmo_array, cosmo_quantity, cosmo_factor
from types import SimpleNamespace

def dict_to_namespace(d):
    """
    Recursively convert nested dicts into SimpleNamespace
    so cfg.section.parameter works again.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d



def cfg_from_serializable(cfg_serialized):
    """
    Recursively convert a serialized cfg (dict from JSON) back to
    cosmo_array, cosmo_quantity, or native Python types with units.
    
    Expects dicts of the form:
        {"value": [...], "unit": "Mpc"}  -> cosmo_array
        {"value": float, "unit": "cm**-2"} -> cosmo_quantity
    """
    if isinstance(cfg_serialized, dict):
        # Check if this dict represents a cosmo_array/quantity
        if "value" in cfg_serialized and "unit" in cfg_serialized:
            val = cfg_serialized["value"]
            unit = u.Unit(cfg_serialized["unit"])
            if isinstance(val, list):
                # It's an array
                return u.unyt_array(val, unit)
            else:
                # Single value
                return u.unyt_quantity(val, unit)
        else:
            # Recurse through dict
            return {k: cfg_from_serializable(v) for k, v in cfg_serialized.items()}

    elif isinstance(cfg_serialized, list):
        return [cfg_from_serializable(v) for v in cfg_serialized]

    # Scalars (float, int, str, bool) are returned as is
    else:
        return cfg_serialized


class unwrapper:

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.galaxy.single_galaxy:
            self.halo_selection_field=self._halo_selection_field()

    # ==========================================================
    # Snapshot
    # ==========================================================

    @property
    def snapshot_path(self) -> str:
        path = (Path(self.cfg.simulation.main_dir) / 
               f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}" / 
               self.cfg.simulation.name / 
               "snapshots" / 
               f"colibre_{self.cfg.simulation.snapshot_number:04d}" / 
               f"colibre_{self.cfg.simulation.snapshot_number:04d}.hdf5"
        )

        path = str(path)

        
        return path
    #build mask + metadata once
    @property
    def mask_snapshot(self):
        return swift.mask(self.snapshot_path)
    

    @property
    def box_size(self):
        return self.mask_snapshot.metadata.boxsize[0]

    @property
    def scale_factor(self):

        return self.mask_snapshot.metadata.scale_factor
    
    @property
    def cosmology(self):

        return self.mask_snapshot.metadata.cosmology
    
    def load_snapshot(self, load_region=None):
        
        if load_region is None:
            return load(self.snapshot_path)
        
        #new mask for spatial constraints
        mask = swift.mask(self.snapshot_path)  # full mask, with gas/dm
        mask.constrain_spatial(load_region)
        return load(self.snapshot_path, mask=mask)

    # ==========================================================
    # Halo properties
    # ==========================================================

    @property
    def halo_properties_path(self) -> str:
        path = (Path(self.cfg.simulation.main_dir) / 
               f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}" / 
               self.cfg.simulation.name / 
               "SOAP-HBT" / 
               f"halo_properties_{self.cfg.simulation.snapshot_number:04d}.hdf5"
        )

        return str(path)

    def load_halo_properties(self):
        return load(self.halo_properties_path)
    
    """
    def _halo_selection_field(self):
        gal_window=self.cfg.galaxy.galaxy_window
        if gal_window in ["inclusive_sphere", "exclusive_sphere"]:
            return f"{gal_window}_{self.cfg.galaxy.extend_value}{self.cfg.galaxy.extend_unit}"
        
        elif gal_window in ["bound_subhalo"]:
            return gal_window
        
        elif gal_window in ["projected_aperture"]:
            return f"{gal_window}_{self.cfg.galaxy.extend_value}{self.cfg.galaxy.extend_unit}_proj{self.cfg.window.projection_axis}"
        else:
            raise ValueError(f"Unknown galaxy window type: {gal_window}")
    """
    def _halo_selection_field(self):
        gal_window=self.cfg.galaxy.galaxy_window
        return gal_window

    # ==========================================================
    # Gas in halo
    # ==========================================================

    @property
    def gas_in_halo_properties_path(self) -> str:
        path =(Path(self.cfg.simulation.main_dir) / 
               f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}" / 
               self.cfg.simulation.name / 
               "SOAP-HBT" / 
               f"colibre_with_SOAP_membership_{self.cfg.simulation.snapshot_number:04d}.hdf5"
        )
        return str(path)

    

    def load_gas_in_halo_properties(self):
        return load(self.gas_in_halo_properties_path)

    # ==========================================================
    # Redshift + snapshot type
    # ==========================================================

    @property
    def redshift_and_type(self):
        path = (Path(self.cfg.simulation.main_dir) / 
               f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}" / 
               self.cfg.simulation.name / 
               "output_list.txt"
        )

        df = pandas.read_csv(
            str(path),
            comment="#",
            header=None,
            names=["redshift", "type"]
        )

        df["type"] = df["type"].str.strip()

        redshift = float(df.iloc[self.cfg.simulation.snapshot_number]["redshift"])
        snapshot_type = df.iloc[self.cfg.simulation.snapshot_number]["type"]

        return redshift, snapshot_type

    @property
    def redshift(self):
        return self.redshift_and_type[0]


    @property
    def snapshot_type(self):
        return self.redshift_and_type[1]

    # ==========================================================
    # CHIMES table
    # ==========================================================

    @property
    def chimes_table_path(self) -> str:
        return str(
            Path(self.cfg.simulation.chimes_table_dir) / 
            f"z{self.redshift:.3f}_eqm.hdf5"
        )

    # ==========================================================
    # Output directory
    # ==========================================================

    @property
    def output_directory(self) -> str:
        path = (Path(self.cfg.data_output.main_dir)/ 
          self.cfg.data_output.results_dir / 
          f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}" / 
          self.cfg.simulation.name /
          f"z_{self.redshift:.3f}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    
### --- WE MIGHT WANNA REDO THE COSMO_ARRAY OVER HERE, NOT QUITE SURE IF I WANT TO STICK TO UNYT RIGHT NOW ---

class single_cd:

    def __init__(self, hdf5_path,load_cd=False):

        self.hdf5_path = hdf5_path
        self.load_cd = load_cd
        

        with h5py.File(hdf5_path, "r") as f:

            # --- read cfg ---
            cfg_serialized = json.loads(f.attrs["cfg"])
            cfg_dict = cfg_from_serializable(cfg_serialized)
            self.cfg = dict_to_namespace(cfg_dict)
            
            # --- read edges ---
            self.xedges = u.unyt_array(
                f["xedges"][:],
                units=f["xedges"].attrs.get("unit", "Mpc")
            )

            self.yedges = u.unyt_array(
                f["yedges"][:],
                units=f["yedges"].attrs.get("unit", "Mpc")
            )

            # --- element name ---
            element = self.cfg.chemistry.element
            self.element_name = element

            grp_elem = f[element]

            # Conditionally load column density
            if load_cd:
                ds_elem = grp_elem["column_density"]

                self.element_cd = u.unyt_array(
                    ds_elem[:],
                    units=ds_elem.attrs.get("unit", "cm**-2")
                )
            else:
                self.element_cd = None

            self.element_cddf = grp_elem["cddf"][:]
            self.element_bin_centers = grp_elem["bin_centers"][:]
            self.element_bin_width = grp_elem["bin_width"] #single constant log spacing of the binwidth

            # --- ions ---
            self.ions = {}

            for ion in self.cfg.chemistry.ion:

                if ion not in f:
                    continue

                grp = f[ion]

                ion_data = {
                    "cddf": grp["cddf"][:],
                    "bin_centers": grp["bin_centers"][:],
                    "bin_width": grp["bin_width"]
                }

                # Conditionally load column density
                if load_cd:
                    ds = grp["column_density"]
                    ion_data["column_density"] = u.unyt_array(
                        ds[:],
                        units=ds.attrs.get("unit", "cm**-2")
                    )
                else:
                    ion_data["column_density"] = None

                self.ions[ion] = ion_data
    @property
    def simulation_name(self):
        feedback = f"{self.cfg.simulation.name}" 
        if "THERMAL_AGN" in feedback:
            feedback = "Thermal"
        elif "HYBRID_AGN" in feedback:
            feedback = "Hybrid"
        return f"L{self.cfg.simulation.box_length:03d}/m{self.cfg.simulation.resolution}/{feedback}"
    




     