from dataclasses import dataclass, field
from typing import List, Union

@dataclass
class Simulation:
    main_dir: str
    chimes_table_dir: str
    box_length: int
    resolution: int
    name: str
    snapshot_number: Union[int, List[int]]  # can be a single int or list

@dataclass
class Data_output:
    main_dir: str
    results_dir:str
    save_projection: bool = True

@dataclass
class Cddf:
    log_range: List[float]
    bins: int


@dataclass
class Monitoring:
    cpu_ram_monitor: bool
    monitor_interval: int

@dataclass
class Window:
    x: List[float]
    y: List[float]
    z: List[float]
    
    resolution: int
    z_center: float
    z_range: List[float]
    #defaulted
    projection_axis: str = "z"
    projection_slices: int =1

@dataclass
class Chemistry:
     element: Union[str, List[str]]
     ion: Union[str, List[str]]
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

  
