from dataclasses import dataclass, field
from typing import List, Union

@dataclass
class Simulation:
    main_dir: str = "/cosma8/data/dp004/colibre/Runs"
    chimes_table_dir: str = "/cosma8/data/dp004/jlvc76/COLIBRE/Tables/equilibrium_rates_and_speciesfractions/abundances_full/"
    box_length: int = 100
    resolution: int = 6
    name: str = "Thermal"
    snapshot_number: Union[int, List[int]] = 127  # can be a single int or list

@dataclass
class Data_output:
    main_dir: str = "/cosma8/data/do012/dc-kort1/CGM/"
    results_dir:str = "results"
    save_projection: bool = False

@dataclass
class Cddf:
    log_range: List[float] = field(default_factory=lambda: [13.0, 20.0])
    bins: int = 1000

@dataclass
class Omega_ion:
    calculate: bool = False



@dataclass
class Monitoring:
    cpu_ram_monitor: bool = False
    monitor_interval: int = 100


@dataclass
class Window:
    x: List[float] = field(default_factory=lambda: [0.0, 1.0])
    y: List[float] = field(default_factory=lambda: [0.0, 1.0])
    z: List[float] = field(default_factory=lambda: [0.0, 1.0])  
    
    resolution: int = 1000
    z_center: float = 1.0
    z_range: List[float] = field(default_factory=lambda: [0.0, 2.0])
    projection_axis: str = "z"
    projection_slices: int =1

@dataclass
class Chemistry:
     element: Union[str, List[str]] = None
     ion: Union[str, List[str]] = None
     metallicity: bool = True

    
@dataclass
class Galaxy:     
     single_galaxy: bool = False 
     galaxy_window: str = None
     extend_unit: float = "kpc"
     extend_value: int = 400
     range_transverse: List[float] = field(default_factory=lambda: [10.0, 300.0])
     bins_transverse: int = 50     
     selection: str = "random " 
     mass_range: List[float] = None
     
     


     
#important class that orders all the configurations
@dataclass
class Config:
    simulation: Simulation
    data_output: Data_output
    window: Window
    chemistry: Chemistry
    galaxy: Galaxy
    monitoring: Monitoring
    cddf: Cddf
    omega_ion: Omega_ion

  
