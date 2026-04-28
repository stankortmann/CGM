import numpy as np
from functools import cached_property
from swiftgalaxy import SWIFTGalaxy, SOAP
import unyt as u
class gas_properties:
    pass


class element_properties:
    pass


class single_galaxy:

    def __init__(self, cfg, data_unpacker):

        self.halo_properties = data_unpacker.load_halo_properties()
        self.gas_in_halo_properties = data_unpacker.load_gas_in_halo_properties()
        #loading the individual gas particles data
        self.gas=self.gas_in_halo_properties.gas
        self.cfg = cfg
        #get the field in which the particles are stored
        self.halo_selection_field=getattr(self.halo_properties,data_unpacker.halo_selection_field)

        #gas particles are now written onto the original self.gas_in_halo_properties.gas:
        self.gas_in_halo_properties.gas=self._retrieve_bound_gas_particles()
       

    # ==========================================================
    # Selection Logic
    # ==========================================================


    def _select_halo(self):
        
        if self.cfg.galaxy.selection == "most_bound_particles":
            bound = self.halo_selection_field.number_of_bound_particles
            idx = np.argmax(bound)

        elif self.cfg.galaxy.selection == "highest_gas_mass":
            gas_mass = self.halo_selection_field.gas_mass
            idx = np.argmax(gas_mass)

        elif self.cfg.galaxy.selection == "random":
            total = len(self.halo_selection_field.gas_mass)
            idx = np.random.randint(0, total)
        
        elif self.cfg.galaxy.selection == "gas_mass_range":
            gas_mass = self.halo_selection_field.gas_mass.to(u.Msun)
            log_gas_mass = np.log10(gas_mass,out=np.zeros_like(gas_mass),where=(gas_mass>0))
            mask = (log_gas_mass >= self.cfg.galaxy.mass_range[0]) & (log_gas_mass <= self.cfg.galaxy.mass_range[1])
            idx = np.where(mask)[0]
            if len(idx) == 0:
                raise ValueError("No halos found in the specified gas mass range")
            idx = np.random.choice(idx)

        elif self.cfg.galaxy.selection == "total_mass":
            total_mass = self.halo_selection_field.total_mass.to(u.Msun)
            log_total_mass = np.log10(total_mass,out=np.zeros_like(total_mass),where=(total_mass>0))
            mask = (log_total_mass >= self.cfg.galaxy.mass_range[0]) & (log_total_mass <= self.cfg.galaxy.mass_range[1])
            idx = np.where(mask)[0]
            if len(idx) == 0:
                raise ValueError("No halos found in the specified total mass range")
            idx = np.random.choice(idx)

        else:
            raise ValueError(f"Unknown selection mode {self.cfg.galaxy.selection}")

        self._retrieve_halo(idx)

    # ==========================================================
    # Halo Properties (lazy)
    # ==========================================================

    def _retrieve_halo(self, index):

        self.catalogue_id =  self.halo_properties.input_halos.halo_catalogue_index[index]

        mask = (
            self.halo_properties.input_halos.halo_catalogue_index
            == self.catalogue_id
        )

        

        self.position = self.halo_selection_field.centre_of_mass[index]
        self.stellar_mass = self.halo_selection_field.stellar_mass[index]
        self.half_mass_radius_gas = self.halo_selection_field.half_mass_radius_gas[index]

        self.halo_mask = mask

    

    def _retrieve_bound_gas_particles(self):
        # select a halo
        self._select_halo()
        
        # precompute indices
        indices = np.nonzero(self.gas.halo_catalogue_index == self.catalogue_id)[0]
        self.gas_mask = np.zeros_like(self.gas.halo_catalogue_index, dtype=bool)
        self.gas_mask[indices] = True

        
        gas = gas_properties()

        # slice arrays using np.take
        gas.temperatures = np.take(self.gas.temperatures, indices,axis=0)
        gas.densities = np.take(self.gas.densities, indices,axis=0)
        gas.masses = np.take(self.gas.masses, indices,axis=0)
        gas.coordinates = np.take(self.gas.coordinates, indices,axis=0)
        #gas.volumes = np.take(self.gas.volumes, indices,axis=0)
        gas.smoothing_lengths = np.take(self.gas.smoothing_lengths, indices,axis=0)
        gas.metal_mass_fractions = np.take(self.gas.metal_mass_fractions, indices,axis=0)

        # element fractions
        elem = element_properties()
        for element in self.gas.element_mass_fractions.named_columns:
            arr = getattr(self.gas.element_mass_fractions, element)
            setattr(elem, element, np.take(arr, indices,axis=0))
        gas.element_mass_fractions = elem

        return gas


class single_galaxy_gas_mask:

    def __init__(self, cfg, data_unpacker):

        self.halo_properties = data_unpacker.load_halo_properties()
        self.cfg = cfg
        #get the field in which the particles are stored
        self.halo_selection_field=getattr(self.halo_properties,data_unpacker.halo_selection_field)

        #loads halo and selects region for which to load in the gas
        self._select_halo()
        #load in the snapshot surrounding the centre of mass of the galaxy/halo
        self.snapshot=data_unpacker.load_snapshot(load_region=self.gas_mask)

       

    # ==========================================================
    # Selection Logic
    # ==========================================================

    
    def _select_halo(self):
        
        if self.cfg.galaxy.selection == "most_bound_particles":
            bound = self.halo_selection_field.number_of_bound_particles
            idx = np.argmax(bound)

        elif self.cfg.galaxy.selection == "highest_gas_mass":
            gas_mass = self.halo_selection_field.gas_mass
            idx = np.argmax(gas_mass)

        elif self.cfg.galaxy.selection == "random":
            total = len(self.halo_selection_field.gas_mass)
            idx = np.random.randint(0, total)

        else:
            raise ValueError(f"Unknown selection mode {self.cfg.galaxy.selection}")

        self._retrieve_halo(idx)

    # ==========================================================
    # Halo Properties (lazy)
    # ==========================================================

    def _retrieve_halo(self, index):

        self.catalogue_id =  self.halo_properties.input_halos.halo_catalogue_index[index]


        self.position = self.halo_selection_field.centre_of_mass[index]
        self.stellar_mass = self.halo_selection_field.stellar_mass[index]
        self.half_mass_radius_gas = self.halo_selection_field.half_mass_radius_gas[index]

        

    #I want it stored and not repeatedly calculated
    @cached_property
    def gas_mask(self):

        barrier = self.cfg.galaxy.extend.to_comoving() # is already a cosmo_quantity
        x0 = self.position[0]
        y0 = self.position[1]
        z0 = self.position[2]
        
        mask=[[x0 - barrier, x0 + barrier],
              [y0 - barrier, y0 + barrier],
              [z0 - barrier, z0 + barrier]]
        
        return mask



#--- USES SWIFTGALAXY TO LOAD IN GAS PARTICLES, MASSIVE I/0 IMPROVEMENT!!---
class single_galaxy_swift_galaxy:

    def __init__(self, cfg, data_unpacker):

        self.halo_properties = data_unpacker.load_halo_properties()
        self.cfg = cfg
        #get the field in which the particles are stored
        self.halo_selection_field=getattr(self.halo_properties,data_unpacker.halo_selection_field)

        #loads halo and selects region for which to load in the gas
        halo_index = int(self._select_halo())
        self._retrieve_halo(halo_index)
        #load in the snapshot surrounding the centre of mass of the galaxy/halo
        self.snapshot=SWIFTGalaxy(
                    data_unpacker.gas_in_halo_properties_path,  # notice virtual_snapshot, not snapshot
                    SOAP(data_unpacker.halo_properties_path, soap_index=halo_index)  
                )

        loaded_n_gas = int(np.asarray(self.snapshot.gas.masses).size)
        expected_n_gas = None
        if hasattr(self.halo_selection_field, "number_of_gas_particles"):
            expected_n_gas = int(np.asarray(self.halo_selection_field.number_of_gas_particles[halo_index]).item())

        print(
            f"Selected halo catalogue_id={int(np.asarray(self.catalogue_id).item())}, "
            f"expected_n_gas={expected_n_gas}, loaded_n_gas={loaded_n_gas}"
        )

        if loaded_n_gas == 0:
            raise ValueError(
                "Selected halo has zero gas particles in SWIFTGalaxy membership load. "
                
            )
       

       

    # ==========================================================
    # Selection Logic
    # ==========================================================

    
    def _select_halo(self):
        n_gas = None
        if hasattr(self.halo_selection_field, "number_of_gas_particles"):
            n_gas = np.asarray(self.halo_selection_field.number_of_gas_particles)
        
        if self.cfg.galaxy.selection == "most_bound_particles":
            bound = self.halo_selection_field.number_of_gas_particles
            idx = np.argmax(bound)

        elif self.cfg.galaxy.selection == "highest_gas_mass":
            gas_mass = self.halo_selection_field.gas_mass
            if n_gas is not None:
                valid_idx = np.where(n_gas > 0)[0]
                if len(valid_idx) == 0:
                    raise ValueError("No halos with number_of_gas_particles > 0 were found.")
                idx = valid_idx[np.argmax(gas_mass[valid_idx])]
            else:
                idx = np.argmax(gas_mass)

        elif self.cfg.galaxy.selection == "random":
            if n_gas is not None:
                valid_idx = np.where(n_gas > 0)[0]
                if len(valid_idx) == 0:
                    raise ValueError("No halos with number_of_gas_particles > 0 were found.")
                idx = np.random.choice(valid_idx)
            else:
                total = len(self.halo_selection_field.gas_mass)
                idx = np.random.randint(0, total)
        
        elif self.cfg.galaxy.selection == "gas_mass_range":
            gas_mass = self.halo_selection_field.gas_mass.to("Msun")
            log_gas_mass = np.log10(gas_mass,out=np.zeros_like(gas_mass),where=(gas_mass>0))
            mask = (log_gas_mass >= self.cfg.galaxy.mass_range[0]) & (log_gas_mass <= self.cfg.galaxy.mass_range[1])
            if n_gas is not None:
                mask = mask & (n_gas > 0)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                raise ValueError("No halos found in the specified gas mass range")
            idx = np.random.choice(idx)
            print(f"Selected halo index {idx} with gas mass {gas_mass[idx]:.2e}")

        elif self.cfg.galaxy.selection == "total_mass":
            total_mass = self.halo_selection_field.total_mass.to("Msun")
            log_total_mass = np.log10(total_mass,out=np.zeros_like(total_mass),where=(total_mass>0))
            mask = (log_total_mass >= self.cfg.galaxy.mass_range[0]) & (log_total_mass <= self.cfg.galaxy.mass_range[1])
            if n_gas is not None:
                mask = mask & (n_gas > 0)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                raise ValueError("No halos found in the specified total mass range")
            idx = np.random.choice(idx)
            print(f"Selected halo index {idx} with total mass {total_mass[idx]:.2e}")

        else:
            raise ValueError(f"Unknown selection mode: {self.cfg.galaxy.selection}")

        return int(np.asarray(idx).item())

    # ==========================================================
    # Halo Properties (lazy)
    # ==========================================================

    def _retrieve_halo(self, index):

        self.catalogue_id =  self.halo_properties.input_halos.halo_catalogue_index[index]


        self.position = self.halo_selection_field.centre_of_mass[index]
        self.stellar_mass = self.halo_selection_field.stellar_mass[index]
        self.half_mass_radius_gas = self.halo_selection_field.half_mass_radius_gas[index]

    @cached_property
    def mask(self):

        barrier = self.cfg.galaxy.extend.to_comoving() # is already a cosmo_quantity
        x0 = self.position[0]
        y0 = self.position[1]
        z0 = self.position[2]
        
        mask=[[x0 - barrier, x0 + barrier],
              [y0 - barrier, y0 + barrier],
              [z0 - barrier, z0 + barrier]]
        
        return mask



        