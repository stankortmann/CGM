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
        self.snapshot_path = self.file_snapshot()
        self.soap_hbt_path = self.file_soap_hbt()
        self.redshift, self.snapshot_type = self.unpack_redshift_type()
        self.chimes_table_path = self.file_chimes_table()

    def file_snapshot(self):
        snapshot_path = str(
                Path(cfg.simulation.main_dir)
                /f"L{cfg.simulation.box_length:03d}_m{cfg.simulation.resolution}"
                /cfg.simulation.name
                /"snapshots"
                /f"colibre_{cfg.simulation.snapshot_number:04d}"
                /f"colibre_{cfg.simulation.snapshot_number:04d}.hdf5"

                )
        
        self.mask_snapshot = swift.mask(snapshot_path)
        # The full metadata object is available from within the mask
        self.box_size = self.mask_snapshot.metadata.boxsize #boxsize
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
    

    def file_soap_hbt(self):
        #open with swiftsimio
        soap_hbt_path = str(
                        Path(cfg.simulation.main_dir)
                        /f"L{cfg.simulation.box_length:03d}_m{cfg.simulation.resolution}"
                        /cfg.simulation.name
                        /"SOAP-HBT"
                        /f"halo_properties_{cfg.simulation.snapshot_number:04d}.hdf5"
                    )
        return soap_hbt_path
    def load_soap_hbt(self):
        #open with swiftsimio
        soap_hbt = load(self.soap_hbt_path)
        return soap_hbt

    def unpack_redshift_type(self):
        #redshift list and output list of the simulation, to be used for loading the correct chimes table
        redshift_list_path= str(Path(cfg.simulation.main_dir)
            /f"L{cfg.simulation.box_length:03d}_m{cfg.simulation.resolution}"
            /cfg.simulation.name
            /"output_list.txt")

        #redshift extraction from the output list, to be used for loading the correct chimes table
        redshift_list = pandas.read_csv(
            redshift_list_path,
            comment="#",
            header=None,
            names=["redshift", "type"]
        )
        redshift_list["type"] = redshift_list["type"].str.strip()
        redshift=float(redshift_list.iloc[cfg.simulation.snapshot_number]["redshift"])
        snapshot_type=redshift_list.iloc[cfg.simulation.snapshot_number]["type"]

        return redshift, snapshot_type
    
    def file_chimes_table(self):
        chimes_table_path = str(
            Path(cfg.simulation.chimes_table_dir)
            /f"z{self.redshift:.3f}_eqm.hdf5"
        )
        return chimes_table_path
    
    def load_chimes_table(self, element):
        
        with h5py.File(self.chimes_table_path, "r") as f:
    
            # 4D abundance grid
            #abundances=(N_Temperatures x N_Densities x N_Metallicities x N_species)
            abundances = f["Abundances"][:]  
            
            
            log_T_table  = f["TableBins/Temperatures"][:]
            log_nH_cm3_table = f["TableBins/Densities"][:]
            log_Z_table  = f["TableBins/Metallicities"][:]
        
        return abundances[:, :, :, element], log_T_table, log_nH_cm3_table, log_Z_table
    
    
