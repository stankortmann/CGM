import numpy as np


class gas_properties:
    pass


class element_properties:
    pass


class single_galaxy:

    def __init__(self, cfg, halo_properties, gas_in_halo_properties):

        self.halo_properties = halo_properties
        self.gas_in_halo_properties = gas_in_halo_properties
        self.cfg = cfg

        # Selection is now lazy
        self._catalogue_id = None
        self._halo_mask = None
        self._gas_mask = None

    # ==========================================================
    # Selection Logic
    # ==========================================================

    @property
    def catalogue_id(self):
        if self._catalogue_id is None:
            self._select_halo()
        return self._catalogue_id

    # ----------------------------------------------------------

    def _select_halo(self):

        if self.cfg.galaxy.selection == "most_bound_particles":
            bound = self.halo_properties.input_halos.number_of_bound_particles
            idx = np.argmax(bound)

        elif self.cfg.galaxy.selection == "highest_gas_mass":
            gas_mass = self.halo_properties.bound_subhalo.gas_mass
            idx = np.argmax(gas_mass)

        elif self.cfg.galaxy.selection == "random":
            total = len(self.halo_properties.bound_subhalo.gas_mass)
            idx = np.random.randint(0, total)

        else:
            raise ValueError("Unknown selection mode")

        self._retrieve_halo(idx)

    # ==========================================================
    # Halo Properties (lazy)
    # ==========================================================

    def _retrieve_halo(self, index):

        self._catalogue_id = self.halo_properties.input_halos.halo_catalogue_index[index]

        mask = (
            self.halo_properties.input_halos.halo_catalogue_index
            == self._catalogue_id
        )

        inclusive = self.halo_properties.inclusive_sphere_50kpc

        self.position = self.halo_properties.input_halos.halo_centre[mask]
        self.stellar_mass = inclusive.stellar_mass[mask]
        self.half_mass_radius_gas = inclusive.half_mass_radius_gas[mask]

        self._halo_mask = mask

   

    @property
    def gas(self):
        if not hasattr(self, "_gas"):
            self._retrieve_bound_gas_particles()
        return self._gas

    

    def _retrieve_bound_gas_particles(self):

        if self._catalogue_id is None:
            self._select_halo()

        mask = (
            self.gas_in_halo_properties.halo_catalogue_index
            == self._catalogue_id
        )

        gas = self.gas_in_halo_properties

        g = gas_properties()
        g.temperatures = gas.temperatures[mask]
        g.metal_mass_fractions = gas.metal_mass_fractions[mask]
        g.densities = gas.densities[mask]
        g.masses = gas.masses[mask]
        g.coordinates = gas.coordinates[mask]

        # ---- element fractions ----
        elem = element_properties()

        for element in gas.element_mass_fractions.named_columns:
            arr = getattr(gas.element_mass_fractions, element)
            setattr(elem, element, arr[mask])

        g.element_mass_fractions = elem

        self._gas = g