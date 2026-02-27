import pandas
import unyt as u
import h5py
import gc
import numpy as np
from pathlib import Path
from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as swift



class unwrapper:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.snapshot_path = self._file_snapshot()
        self.halo_properties_path = self._file_halo_properties()
        self.gas_in_halo_properties_path=self._file_gas_in_halo_properties()
        self.redshift, self.snapshot_type = self._unpack_redshift_type()
        self.chimes_table_path = self._file_chimes_table()

    def _file_snapshot(self):
        snapshot_path = str(
                Path(self.cfg.simulation.main_dir)
                /f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}"
                /self.cfg.simulation.name
                /"snapshots"
                /f"colibre_{self.cfg.simulation.snapshot_number:04d}"
                /f"colibre_{self.cfg.simulation.snapshot_number:04d}.hdf5"

                )
        
        self.mask_snapshot = swift.mask(snapshot_path)
        # The full metadata object is available from within the mask
        self.box_size = self.mask_snapshot.metadata.boxsize[0] #boxsize in physical units  
        return snapshot_path
    
    def load_snapshot(self, load_region=None):
        #no constraint on loading region, load the whole snapshot
        if load_region is None:
            snapshot = load(self.snapshot_path)
        else:
            # Constrain the mask
            self.mask_snapshot.constrain_spatial(load_region)
            # Now load the snapshot with this mask
            snapshot = load(self.snapshot_path, mask=self.mask_snapshot)
        return snapshot
    

    def _file_halo_properties(self):
        #open with swiftsimio
        halo_properties_path = str(
                        Path(self.cfg.simulation.main_dir)
                        /f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}"
                        /self.cfg.simulation.name
                        /"SOAP-HBT"
                        /f"halo_properties_{self.cfg.simulation.snapshot_number:04d}.hdf5"
                    )
        return halo_properties_path
    def load_halo_properties(self):
        #open with swiftsimio
        halo_properties = load(self.halo_properties_path)
        return halo_properties

    def _file_gas_in_halo_properties(self):
        #open with swiftsimio
        gas_in_halo_properties_path = str(
                        Path(self.cfg.simulation.main_dir)
                        /f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}"
                        /self.cfg.simulation.name
                        /"SOAP-HBT"
                        /f"colibre_with_SOAP_membership_{self.cfg.simulation.snapshot_number:04d}.hdf5"
                    )
        return  gas_in_halo_properties_path
    
    def load_gas_in_halo_properties(self):
        #open with swiftsimio
        gas_in_halo_properties=load(self.gas_in_halo_properties_path)
            
        return gas_in_halo_properties

    def _unpack_redshift_type(self):
        #redshift list and output list of the simulation, to be used for loading the correct chimes table
        redshift_list_path= str(Path(self.cfg.simulation.main_dir)
            /f"L{self.cfg.simulation.box_length:03d}_m{self.cfg.simulation.resolution}"
            /self.cfg.simulation.name
            /"output_list.txt")

        #redshift extraction from the output list, to be used for loading the correct chimes table
        redshift_list = pandas.read_csv(
            redshift_list_path,
            comment="#",
            header=None,
            names=["redshift", "type"]
        )
        redshift_list["type"] = redshift_list["type"].str.strip()
        redshift=float(redshift_list.iloc[self.cfg.simulation.snapshot_number]["redshift"])
        snapshot_type=redshift_list.iloc[self.cfg.simulation.snapshot_number]["type"]

        return redshift, snapshot_type
    
    def _file_chimes_table(self):
        chimes_table_path = str(
            Path(self.cfg.simulation.chimes_table_dir)
            /f"z{self.redshift:.3f}_eqm.hdf5"
        )
        return chimes_table_path
    
    
