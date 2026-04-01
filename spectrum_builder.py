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


fontsize= 20
ix           = Wizard['sightline']["x-axis"] ; iy = Wizard['sightline']["y-axis"] ; iz = Wizard['sightline']["z-axis"]
positions    = snap_data["Particles"]["Positions"]
densities    = snap_data["Particles"]["Densities"]
elementnames = Wizard["ionparams"]["Ions"]

zpos = positions["Value"][:,iz]
rho  = densities["Value"]

# parameters of projection
pixz = projected_data["pixel"]["Value"] * np.arange(projected_data["npix"])

# Save figures
out_dir = "/cosma8/data/do012/dc-kort1/CGM/short_spectra"
os.makedirs(out_dir, exist_ok=True)


def _safe_name(name):
    return name.replace(" ", "_").replace("/", "_")


los_file = os.path.basename(Wizard.get("snapshot_params", {}).get("file", "unknown_los"))
los_file_tag = _safe_name(os.path.splitext(los_file)[0])
los_num = Wizard.get("sightline", {}).get("nsight", "unknown")
los_dir = os.path.join(out_dir, los_file_tag, f"los_{los_num}")
os.makedirs(los_dir, exist_ok=True)


for element, ion in elementnames:
    OD = optical_depth[(element, ion)]["Optical depths"]["Value"]
    print("printing the column density for the ions")
    Nion = optical_depth[(element, ion)]["TotalIonColumnDensity"]["Value"]
    print(element, ion, np.log10(Nion))
    element_density = projected_data["Element-weighted"][element]["Densities"]["Value"]
    ion_density = projected_data["Ion-weighted"][ion]["Densities"]["Value"]
    ion_mass = Wizard["ionparams"]["transitionparams"][ion]["Mass"] * constants["amu"]
    ion_number_density = (ion_density / ion_mass).in_cgs()


    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(element_density > 0, ion_density / element_density, np.nan)
        log_ratio = np.log10(ratio)
        log_ion_number_density = np.log10(np.where(ion_number_density.value > 0, ion_number_density.value, np.nan))
    redshift = str(round(Wizard["Header"]["Cosmo"]["Redshift"],2))
    fig, ax = plt.subplots(4, 1, figsize=(20, 16))

    ax[0].plot(OD, color="k")
    ax[0].set_title(f"{element} {ion} at redshift {redshift}", fontsize=fontsize)
    ax[0].set_ylabel(r"$\tau$", fontsize=fontsize)
    ax[0].set_xlabel(r"Velocity [km/s]", fontsize=fontsize)

    transmission = np.exp(-OD)
    ax[1].plot(transmission, color="tab:green")
    ax[1].set_ylabel(rf"$\exp(-\tau)$", fontsize=fontsize)
    ax[1].set_xlabel(r"Velocity [km/s]", fontsize=fontsize)

    ax[2].plot(pixz, log_ratio, color="tab:blue")
    ax[2].set_ylabel(rf"$\log_{{10}}(n_{{ion}}/n_{{element}})$", fontsize=fontsize)
    ax[2].set_xlabel(r"$z$ [Mpc]", fontsize=fontsize)

    ax[3].plot(pixz, log_ion_number_density, color="tab:purple")
    ax[3].set_ylabel(r"$\log_{10}(n_{\rm ion}\,[{\rm cm}^{-3}])$", fontsize=fontsize)
    ax[3].set_xlabel(r"$z$ [Mpc]", fontsize=fontsize)
    
    

    fig.tight_layout()

    ion_tag = _safe_name(ion)
    element_tag = _safe_name(element)
    out_file = os.path.join(los_dir, f"{element_tag}/{ion_tag}/data.png")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_file}")


## -creation of full transmission plot for all ions together
Numof_Ions = len(elementnames)
fig, ax = plt.subplots(Numof_Ions, 1, figsize=(20, max(10, 3 * Numof_Ions)))

if Numof_Ions == 1:
    ax = [ax]

for i, (element, ion_name) in enumerate(elementnames):
    try:
        OD = optical_depth[(element, ion_name)]["Optical depths"]["Value"]
    except KeyError:
        continue

    transmission = np.exp(-OD)
    ax[i].plot(transmission, color="k")
    ax[i].set_title(f"{element} {ion_name}")
    ax[i].set_ylabel(r"$\exp(-\tau)$")
    ax[i].set_xlabel(r"$v\ [{\rm km\ s}^{-1}]$")

fig.tight_layout()
transmission_file = os.path.join(los_dir, "all_ions_transmission.png")
fig.savefig(transmission_file, dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(Numof_Ions, 1, figsize=(20, max(10, 3 * Numof_Ions)))
for i, (element, ion_name) in enumerate(elementnames):
    try:
        OD = optical_depth[(element, ion_name)]["Optical depths"]["Value"]
    except KeyError:
        continue
    
    ax[i].plot(OD, color="k")
    ax[i].set_title(f"{element} {ion_name}")
    ax[i].set_ylabel(r"$\tau$")
    ax[i].set_xlabel(r"$v\ [{\rm km\ s}^{-1}]$")

fig.tight_layout()

tau_file = os.path.join(los_dir, "all_ions_tau.png")
fig.savefig(tau_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot to: {tau_file}")
