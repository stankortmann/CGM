import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_array

print("=== STEP 1: Create original cosmo_array ===")

original = cosmo_array(
    np.arange(16).reshape(4, 4),
    units=u.cm**-2,
    comoving=False,
    scale_factor=0.5,
    scale_exponent=-2,
)
print(dir(original.cosmo_factor))
print(original.cosmo_factor)
print("Type:", type(original))
print("Units:", original.units)
print("Comoving:", original.comoving)
print("Data:\n", original)

# -------------------------------------------------
# STEP 2: Simulate MPI tiling
# -------------------------------------------------

print("\n=== STEP 2: Create tiles (simulate MPI ranks) ===")

tiles = [
    original[:2, :2],  # rank 0
    original[:2, 2:],  # rank 1
    original[2:, :2],  # rank 2
    original[2:, 2:],  # rank 3
]

for i, t in enumerate(tiles):
    print(f"\nTile {i} type:", type(t))
    print(t)

# -------------------------------------------------
# STEP 3: Simulate MPI gather
# -------------------------------------------------

print("\n=== STEP 3: Gather tiles (like MPI) ===")

gathered = tiles  # in real MPI: comm.gather(...)
print("Number of gathered tiles:", len(gathered))

# -------------------------------------------------
# STEP 4: Stitch tiles (YOUR CURRENT APPROACH)
# -------------------------------------------------

print("\n=== STEP 4: Stitch into NumPy array ===")

full = np.zeros((4, 4))

full[:2, :2] = gathered[0]
full[:2, 2:] = gathered[1]
full[2:, :2] = gathered[2]
full[2:, 2:] = gathered[3]

print("Type after stitching:", type(full))
print("Data:\n", full)

# Check metadata loss
print("\n--- Checking metadata ---")
try:
    print("Units:", full.units)
except AttributeError:
    print("❌ Units LOST")

# Check values are still correct
print("\nValues preserved:", np.allclose(full, original.value))

# -------------------------------------------------
# STEP 5: Proper reconstruction
# -------------------------------------------------

print("\n=== STEP 5: Reconstruct cosmo_array ===")

reconstructed = cosmo_array(
    full,
    units=original.units,
    comoving=original.comoving,
    cosmo_factor=original.cosmo_factor,
)

print("Type:", type(reconstructed))
print("Units:", reconstructed.units)
print("Comoving:", reconstructed.comoving)
print("Data:\n", reconstructed)

# -------------------------------------------------
# STEP 6: Validate correctness
# -------------------------------------------------

print("\n=== STEP 6: Validation ===")

print("Values match original:",
      np.allclose(reconstructed.value, original.value))

print("Units match:",
      reconstructed.units == original.units)

print("Comoving flag match:",
      reconstructed.comoving == original.comoving)