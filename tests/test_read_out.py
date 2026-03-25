# test_read_out.py

import sys
from pathlib import Path
import numpy as np
from spec_analysis.unpack_data import single_cd

def print_unyt_array(name, arr):
    """Print value, unit, and shape for a unyt array/quantity."""
    print(f"{name}:")
    print(f"  type : {type(arr)}")
    print(f"  shape: {getattr(arr, 'shape', 'scalar')}")
    print(f"  unit : {getattr(arr, 'units', 'dimensionless')}")
    print(f"  value: {arr}")
    print()

def print_diff_stats(name, arr):
    """Print min/max/mean/std for diff arrays."""
    vals = np.asarray(arr)
    unit = getattr(arr, "units", "")
    print(f"{name}:")
    print(f"  shape: {vals.shape}")
    print(f"  min : {vals.min()} {unit}")
    print(f"  max : {vals.max()} {unit}")
    print(f"  mean: {vals.mean()} {unit}")
    print(f"  std : {vals.std()} {unit}")
    print()

def load_cd(hdf5_path):
    hdf5_file = Path(hdf5_path)
    if not hdf5_file.exists():
        raise FileNotFoundError(f"File not found: {hdf5_file}")
    return single_cd(hdf5_file)

def compare_two(file_a, file_b):
    cd_a = load_cd(file_a)
    cd_b = load_cd(file_b)

    print("=== Comparing files ===")
    print(f"A: {file_a}")
    print(f"B: {file_b}")
    print()

    # Optional grid checks
    try:
        same_x = np.allclose(np.asarray(cd_a.xedges), np.asarray(cd_b.xedges))
        same_y = np.allclose(np.asarray(cd_a.yedges), np.asarray(cd_b.yedges))
        print(f"Same xedges: {same_x}")
        print(f"Same yedges: {same_y}")
        print()
    except Exception:
        print("Could not compare x/y edges directly.\n")

    # element column density: A - B
    if cd_a.element_cd.shape != cd_b.element_cd.shape:
        print("element_cd shape mismatch:")
        print(f"  A: {cd_a.element_cd.shape}")
        print(f"  B: {cd_b.element_cd.shape}")
        print()
    else:
        element_cd_diff = cd_a.element_cd - cd_b.element_cd
        print_diff_stats("element_cd (A - B)", element_cd_diff)
        # Check where differences are largest

        max_idx = np.unravel_index(np.argmax(np.abs(np.asarray(element_cd_diff))), element_cd_diff.shape)
        print(f"Largest diff at pixel {max_idx}: {element_cd_diff[max_idx]}")
        print(f"  A[{max_idx}] = {cd_a.element_cd[max_idx]}")
        print(f"  B[{max_idx}] = {cd_b.element_cd[max_idx]}")
  


    # element CDDF: A - B
    if cd_a.element_cddf.shape != cd_b.element_cddf.shape:
        print("element_cddf shape mismatch:")
        print(f"  A: {cd_a.element_cddf.shape}")
        print(f"  B: {cd_b.element_cddf.shape}")
        print()
    else:
        cddf_diff = cd_a.element_cddf - cd_b.element_cddf
        print_diff_stats("element_cddf (A - B)", cddf_diff)

if __name__ == "__main__":
    default_a = "/cosma8/data/do012/dc-kort1/CGM/pixel_size/mid/L025_m6/Thermal/z_0.000/hdf5_data/hydrogen_column_density.hdf5"
    default_b = "/cosma8/data/do012/dc-kort1/CGM/pixel_size/mid_test/L025_m6/Thermal/z_0.000/hdf5_data/hydrogen_column_density.hdf5"

    if len(sys.argv) == 3:
        compare_two(sys.argv[1], sys.argv[2])
    else:
        compare_two(default_a, default_b)