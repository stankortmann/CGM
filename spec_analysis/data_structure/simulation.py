from dataclasses import dataclass
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
    projection_axis: str
    projection_slices: int
    resolution: int
    z_center: float
    z_range: List[float]

@dataclass
class Chemistry:
     element: Union[str, List[str]]
     ion: Union[str, List[str]]
    
@dataclass
class Galaxy:
     single_galaxy: bool 
     galaxy_window: str
     extend_unit: float
     extend_value: int
     selection: str

     
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

  
