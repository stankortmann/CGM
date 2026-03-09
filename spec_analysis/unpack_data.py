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

        #build mask + metadata once
        if not hasattr(self, "_mask_snapshot"):
            self._mask_snapshot = swift.mask(path)
        return path
    

    @property
    def box_size(self):
        # Ensure mask exists by accesssing the path
        _ = self.snapshot_path

        return self._mask_snapshot.metadata.boxsize[0]
    
    def load_snapshot(self, load_region=None):
        if load_region is None:
            return load(self.snapshot_path)

        self._mask_snapshot.constrain_spatial(load_region)
        return load(self.snapshot_path, mask=self._mask_snapshot)

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
    def scale_factor(self):
        return 1/(self.redshift+1)

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
          self.cfg.simulation.name
        )
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    
    
