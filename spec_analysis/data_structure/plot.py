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
    output_directory: str                  # Path where plots will be saved inside the data_directory, so an extra folder will be created
    hdf5_files: Optional[List[str]] = field(default=None)  # Optional list of HDF5 files

    def validate_paths(self):
        """
        Ensure directories exist or create output directory.
        """
        data_dir_path = Path(self.data_directory)
        if not data_dir_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_directory}")

        output_dir_path = Path(self.output_directory)
        output_dir_path.mkdir(parents=True, exist_ok=True)

    def get_hdf5_files(self) -> List[Path]:
        """
        Return a list of HDF5 files to plot.
        If hdf5_files is set, use those, else scan the data_directory.
        """
        if self.hdf5_files:
            # Use the provided filenames
            files = [Path(f) for f in self.hdf5_files]
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