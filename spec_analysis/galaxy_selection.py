import numpy as np


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

        else:
            raise ValueError("Unknown selection mode")

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
        indices = np.nonzero(self.gas_in_halo_properties.halo_catalogue_index == self.catalogue_id)[0]
        self.gas_mask = np.zeros_like(self.gas_in_halo_properties.halo_catalogue_index, dtype=bool)
        self.gas_mask[indices] = True

        gas_all = self.gas_in_halo_properties.gas
        gas = gas_properties()

        # slice arrays using np.take
        gas.temperatures = np.take(gas_all.temperatures, indices,axis=0)
        gas.densities = np.take(gas_all.densities, indices,axis=0)
        gas.masses = np.take(gas_all.masses, indices,axis=0)
        gas.coordinates = np.take(gas_all.coordinates, indices,axis=0)
        gas.volumes = np.take(gas_all.volumes, indices,axis=0)
        gas.smoothing_lengths = np.take(gas_all.smoothing_lengths, indices,axis=0)
        gas.metal_mass_fractions = np.take(gas_all.metal_mass_fractions, indices,axis=0)

        # element fractions
        elem = element_properties()
        for element in gas_all.element_mass_fractions.named_columns:
            arr = getattr(gas_all.element_mass_fractions, element)
            setattr(elem, element, np.take(arr, indices,axis=0))
        gas.element_mass_fractions = elem

        return gas

        