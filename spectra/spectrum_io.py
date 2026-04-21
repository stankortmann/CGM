import copy
import os

import h5py
import numpy as np
import specwizard


class ShortSpectraWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    @staticmethod
    def los_tag(wizard):
        snapshot_cfg = wizard.get("snapshot_params", {}) if isinstance(wizard, dict) else {}
        file_name = os.path.basename(snapshot_cfg.get("file", "los_0000.hdf5"))
        stem, _ = os.path.splitext(file_name)
        if stem.startswith("los_"):
            try:
                return f"LOS_{int(stem.split('_', 1)[1])}"
            except (IndexError, ValueError):
                pass
        return stem.replace("los_", "LOS_").replace("los", "LOS")

    def write_opticaldepth(self, wizard, projected_data, optical_depth):
        nsight = wizard.get("sightline", {}).get("nsight", 0)
        hdf5_dir = os.path.join(self.output_dir, self.los_tag(wizard), "hdf5","")
        os.makedirs(hdf5_dir, exist_ok=True)
        fname = f"los_{nsight}.hdf5"

        save_wizard = copy.deepcopy(wizard)
        save_wizard.setdefault("Output", {})
        save_wizard["Output"]["directory"] = hdf5_dir
        save_wizard["Output"]["fname"] = fname

        writer = specwizard.OpticalDepth_IO(save_wizard, create=True)
        projections = {
            "nsight": nsight,
            "Projection": projected_data,
            "OpticaldepthWeighted": optical_depth,
        }
        writer.write_shortspectra_to_file(projections)
        return os.path.join(hdf5_dir, fname)


class LongSpectraWriter:
    @staticmethod
    def random_hdf5_path(run_output_dir, prefix="long_spectra"):
        hdf5_dir = os.path.join(run_output_dir, "hdf5")
        os.makedirs(hdf5_dir, exist_ok=True)
        fname = f"{prefix}.hdf5"
        return hdf5_dir, fname


    def write_opticaldepth(self, projections_list, run_output_dir, save_basename):
        if len(projections_list) == 0:
            return None

        hdf5_dir, fname = self.random_hdf5_path(run_output_dir, prefix=save_basename)
        writer_wizard = copy.deepcopy(projections_list[0]["wizard"])
        writer_wizard.setdefault("Output", {})
        writer_wizard["Output"]["directory"] = hdf5_dir
        writer_wizard["Output"]["fname"] = fname

        writer = specwizard.OpticalDepth_IO(writer_wizard, create=True)

        for projections in projections_list:
            writer.write_longspectra_to_file(projections)

        hdf5_file = os.path.join(hdf5_dir, fname)
        print(f"Saved long-spectra optical-depth hdf5 to: {hdf5_file}")
        return hdf5_file


class SpectraReader:
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def safe_name(name):
        return str(name).replace(" ", "_").replace("/", "_")

    def default_plot_path(self, nsight, element, ion, kind):
        out_dir = os.path.dirname(self.file_path)
        base = f"{kind}_LOS_{nsight}_{self.safe_name(element)}_{self.safe_name(ion)}.png"
        return os.path.join(out_dir, base)

    def list_sightlines(self):
        with h5py.File(self.file_path, "r") as hfile:
            return sorted([name for name in hfile.keys() if name.startswith("LOS_")])

    def list_ions(self, nsight):
        los = f"LOS_{nsight}"
        ions = []
        with h5py.File(self.file_path, "r") as hfile:
            if los not in hfile:
                return ions
            for element in hfile[los].keys():
                if element == "Box_kms":
                    continue
                for ion in hfile[f"{los}/{element}"].keys():
                    if ion.endswith("-weighted"):
                        continue
                    ions.append((element, ion))
        return ions

    def read_dataset(self, hfile, path):
        if path not in hfile:
            return None
        return np.asarray(hfile[path][...])

    def read_ion_spectrum(self, nsight, element, ion, weighted_group="optical depth-weighted"):
        los = f"LOS_{nsight}"
        if weighted_group == "Element-weighted":
            base = f"{los}/{element}/{weighted_group}"
        else:
            base = f"{los}/{element}/{ion}/{weighted_group}"
        with h5py.File(self.file_path, "r") as hfile:
            return {
                "pixel_kms": self.read_dataset(hfile, f"{base}/pixel_kms"),
                "velocities": self.read_dataset(hfile, f"{base}/Velocities"),
                "densities": self.read_dataset(hfile, f"{base}/Densities"),
                "temperatures": self.read_dataset(hfile, f"{base}/Temperatures"),
                "optical_depths": self.read_dataset(hfile, f"{base}/Optical depths"),
                "total_ion_column_density": self.read_dataset(hfile, f"{base}/TotalIonColumnDensity"),
                "lambda0": self.read_dataset(hfile, f"{base}/lambda0"),
                "f_value": self.read_dataset(hfile, f"{base}/f-value"),
            }

    
