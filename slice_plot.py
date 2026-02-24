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

catalog=load(soap_hbt_path)
positions_haloes=catalog.inclusive_sphere_50kpc.centre_of_mass
#making the same cut in z as for the snapshot
positions_haloes=positions_haloes[(positions_haloes[:,2]>zmin) & (positions_haloes[:,2]<zmax)]

# Project density (mass) along one axis (say z→ surface density on x-y)
surface_density = project_gas(
    snapshot,
    resolution=int(b.value[0])*50,
    
    project="masses",
    periodic=True
)

###### without haloes ######
plt.figure(figsize=(8,8))
plt.imshow(
    surface_density.to_physical_value("Msun/Mpc**2").T,  # convert if units are comoving
    origin="lower",
    norm=LogNorm(),
    cmap="inferno"
)
plt.colorbar(label=f"Surface density [{str(surface_density.units)}]")

plt.xlabel("x [Mpc comoving]")
plt.ylabel("y [Mpc comoving]")
plt.title("Projected surface density (along z)")
plt.tight_layout()
plt.savefig("test_colibre/without_haloes.png", dpi=300)
print("done without haloes")
plt.close()




###### with haloes ######
plt.figure(figsize=(8,8))
plt.imshow(
    surface_density.to_physical_value("Msun/Mpc**2").T,  # convert if units are comoving
    origin="lower",
    norm=LogNorm(),
    cmap="inferno",
    #extent = [load_region[0][0], load_region[0][1],
    #     load_region[1][0], load_region[1][1]]  # x_min, x_max, y_min, y_max
)
plt.colorbar(label=f"Surface density [{str(surface_density.units)}]")
plt.scatter(positions_haloes[:,0],positions_haloes[:,1],s=0.5,c='green',alpha=0.5,label="Halo centres")
plt.xlabel("x [Mpc comoving]")
plt.ylabel("y [Mpc comoving]")
plt.title("Projected surface density (along z)")
plt.tight_layout()
plt.savefig("test_colibre/with_haloes.png", dpi=300)
print("done with haloes")
plt.close()



###### with haloes ######
plt.figure(figsize=(8,8))

plt.scatter(positions_haloes[:,0],positions_haloes[:,1],s=1,c='green',alpha=0.5,label="Halo centres")
plt.xlabel("x [Mpc comoving]")
plt.ylabel("y [Mpc comoving]")
plt.title("Halo centre of mass positions")
plt.tight_layout()
plt.savefig("test_colibre/only_haloes.png", dpi=300)
print("done only haloes")
plt.close()

