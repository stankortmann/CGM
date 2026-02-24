from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as sw
from swiftsimio.visualisation.projection import project_gas
import h5py 
import gc
import numpy as np




import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Load snapshot
snapshot_path = "/cosma8/data/dp004/colibre/Runs/L012_m6/Thermal/snapshots/colibre_0127/colibre_0127.hdf5"
#snapshot_path="/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL/snapshots/flamingo_0000/flamingo_0000.0.hdf5"
soap_hbt_path="/cosma8/data/dp004/colibre/Runs/L012_m6/Thermal/SOAP-HBT/halo_properties_0127.hdf5"

mask = sw.mask(snapshot_path)
# The full metadata object is available from within the mask
b = mask.metadata.boxsize #boxsize
# load_region is a 3x2 list [[left, right], [bottom, top], [front, back]]
zmin = 0.6 * b[2]
zmax = 0.9 * b[2]
#load_region = [[0.0 * b, 0.5 * b] for b in boxsize]
load_region = [
        [0.0 * b[0], 1.0 * b[0]],
        [0.0 * b[1], 1.0 * b[1]],
        [zmin, zmax]
    ]

# Constrain the mask
mask.constrain_spatial(load_region)



# Now load the snapshot with this mask
snapshot = load(snapshot_path, mask=mask)



print(np.shape(snapshot.gas.densities))
print((snapshot.gas.densities[0:5]))




