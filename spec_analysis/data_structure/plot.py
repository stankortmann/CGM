# data_structure_plot.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class plot_config:
    """
    Data structure for plotting configuration.
    """
    label_criterion: str                   # Label or name for this plot / simulation
    data_directory: str                    # Path to the directory containing HDF5 files
    length_unit: str                       # Unit for length (e.g., "Mpc", "kpc", "cm")
    cd_log_range: List[float]             # Range of column density in log scale for 2D histogram [min, max]
    output_directory: str                  # Path where plots will be saved inside the data_directory, so an extra folder will be created
    hdf5_files: Optional[List[str]] = field(default=None)  # Optional list of HDF5 files
    plot_eagle: bool = False               # Whether to plot EAGLE CDDF for comparison
    eagle_cddf_directory: Optional[str] = None  # Path to the directory containing EAGLE CDDF files

    def validate_paths(self):
        """
        Ensure directories exist or create output directory.
        """
        data_dir_path = Path(self.data_directory)
        eagle_dir_path = Path(self.eagle_cddf_directory) if self.eagle_cddf_directory else None
        if not data_dir_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_directory}")
        if self.plot_eagle and (eagle_dir_path is None or not eagle_dir_path.exists()):
            raise FileNotFoundError(f"EAGLE CDDF directory does not exist: {self.eagle_cddf_directory}")
        #create the output directory if it does not exist
        output_dir_path = data_dir_path / self.output_directory
        output_dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def data_files(self) -> List[Path]:
        """
        Return a list of HDF5 files to plot.
        If hdf5_files is set, use those, else scan the data_directory.
        """
        if self.hdf5_files:
            # Use the provided filenames
            files = [Path(self.data_directory)/Path(self.output_directory) / f for f in self.hdf5_files]
            for f in files:
                if not f.exists():
                    raise FileNotFoundError(f"HDF5 file does not exist: {f}")
            return files
        else:
            # Scan the directory
            files = sorted(Path(self.data_directory).glob("*.hdf5"))
            if not files:
                raise FileNotFoundError(f"No HDF5 files found in {self.data_directory}")
            return files