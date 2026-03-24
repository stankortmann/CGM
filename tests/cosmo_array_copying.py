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

copy=original
print(copy)
print("=== STEP 2: Modify the copy ===")
original=5
print(copy)