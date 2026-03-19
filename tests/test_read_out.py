# test_read_out.py

import sys
from pathlib import Path
import unyt as u
from spec_analysis.unpack_data import single_cd

def print_unyt_array(name, arr):
    """Print value, unit, and shape for a unyt array/quantity."""
    print(f"{name}:")
    print(f"  type : {type(arr)}")
    print(f"  shape: {getattr(arr, 'shape', 'scalar')}")
    print(f"  unit : {arr.units}")
    print(f"  value: {arr}")
    print()

def main(hdf5_path):
    hdf5_file = Path(hdf5_path)
    if not hdf5_file.exists():
        print(f"File not found: {hdf5_file}")
        return

    # --- Load HDF5 via single_cd ---
    cd_data = single_cd(hdf5_file)

    # --- Print configuration ---
    print("=== Configuration ===")
    for k, v in vars(cd_data.cfg).items():
        print(f"{k}: {v}")
    print()
    # --- Print edges ---
    print_unyt_array("xedges", cd_data.xedges)
    print_unyt_array("yedges", cd_data.yedges)

    # --- Print element ---
    print_unyt_array(f"Element column_density ({cd_data.element_name})", cd_data.element_cd)
    print(f"Element CDDF shape: {cd_data.element_cddf.shape}")
    print(f"Element bin_centers shape: {cd_data.element_bin_centers.shape}")
    #print(f"Element bin_width shape: {cd_data.element_bin_width.shape}")
    print()

    # --- Print ions ---
    for ion, ion_data in cd_data.ions.items():
        print(f"--- Ion: {ion} ---")
        print_unyt_array(f"{ion} column_density", ion_data["column_density"])
        print(f"{ion} CDDF shape: {ion_data['cddf'].shape}")
        print(f"{ion} bin_centers shape: {ion_data['bin_centers'].shape}")
        #print(f"{ion} bin_width shape: {ion_data['bin_width'].shape}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_read_out.py <path_to_hdf5_file>")
    else:
        main(sys.argv[1])