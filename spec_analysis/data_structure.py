from dataclasses import dataclass
from typing import List, Union

@dataclass
class Simulation:
    main_dir: str
    box_length: int
    resolution: int
    name: str
    snapshot_number: Union[int, List[int]]  # can be a single int or list

@dataclass
class Data_output:
    main_dir: str
    results_dir:str


@dataclass
class Monitoring:
    cpu_ram_monitor: bool
    monitor_interval: int

#important class that orders all the configurations
@dataclass
class Config:
    simulation: Simulation
    data_output: Data_output
    monitoring: Monitoring
  
