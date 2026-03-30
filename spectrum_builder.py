import specwizard
import numpy as np
from specwizard import Phys
import matplotlib.pyplot as plt
import os

constants = Phys.ReadPhys()

build_input = specwizard.Build_Input()

Wizard = build_input.read_from_yml(
    yml_file='/cosma/home/do012/dc-kort1/CGM/configurations/specwizard/template/spec.yaml'
    )

optical_depth,projected_data,snap_data = specwizard.GenerateShortSpectra(Wizard=Wizard)


positions         = snap_data["Particles"]["Positions"] # We load the positions from the simulation data LOS 
densities         = snap_data["Particles"]["Densities"] # We do the same for the density 
projected_density = projected_data["Mass-weighted"]['Densities']['Value'] # We load the Mass-weighted density from the projected pixel data. 
OD           = optical_depth[('Hydrogen', 'H I')]['Optical depths']['Value'] # We load the Optical depth
elementnames = Wizard["ionparams"]["Ions"] # We get the name of the ions that we used


fontsize= 20
ix           = Wizard['sightline']["x-axis"] ; iy = Wizard['sightline']["y-axis"] ; iz = Wizard['sightline']["z-axis"]
positions    = snap_data["Particles"]["Positions"]
densities    = snap_data["Particles"]["Densities"]
OD           = optical_depth[('Hydrogen', 'H I')]['Optical depths']['Value']
elementnames = Wizard["ionparams"]["Ions"]
#
zpos = positions["Value"][:,iz]
rho  = densities["Value"]

# parameters of projection
pixz = projected_data["pixel"]["Value"] * np.arange(projected_data["npix"])
                    
fig, ax = plt.subplots(2, 1, figsize=(20, 12))

title = r"$z=$"+str(round(Wizard["Header"]["Cosmo"]["Redshift"],2))

# plot particle density
ax[0].set_title(title, fontsize=fontsize)
ax[0].plot(zpos, np.log10(rho), ',', label='SPH')
ax[0].plot(pixz, np.log10(projected_data["Mass-weighted"]['Densities']['Value']), color='r', label='Projected (Mass Weighted')
ax[0].legend(frameon=False)
ax[0].set_ylabel(r"$\log\rho$", fontsize=fontsize)
ax[0].set_xlabel(r"$z$", fontsize=fontsize)



ax[1].plot(OD, label=r'$\tau_{\mathrm{H I}}$')

ax[1].legend(frameon=False, fontsize=fontsize)



ax[1].set_ylabel(r"$\tau$", fontsize=fontsize)
ax[1].set_xlabel(r"Velocity [km/s]", fontsize=fontsize)

# Save figure
out_dir = "/cosma8/data/do012/dc-kort1/CGM/short_spectra"
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(
    out_dir,
    f"test.png"
)

fig.tight_layout()
fig.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {out_file}")

fig.close()