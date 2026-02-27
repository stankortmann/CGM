class single_galaxy:

    def __init__(self,cfg,halo_properties, gas_in_halo_properties):
        
        #properties of the halo as a whole
        self.halo_properties=halo_properties
        #properties of bound gas particles in halos
        self.gas_in_halo_properties=gas_in_halo_properties
        self.cfg=cfg
        if cfg.galaxy.selection == "most_bound_particles":
            self._most_bound_particles() 

        if cfg.galaxy.selection == "highest_gas_mass":
            self._highest_gas_mass_halo()   

    def _retrieve_bound_gas_particles(self):
        
        gas_particles_in_halo_mask=(self.gas_in_halo_properties.halo_catalogue_index==self.halo_catalogue_id)
        self.gas=self.gas_in_halo_properties[gas_particles_in_halo]
        
        return 

    def _retrieve_halo(self,index):
        #for now we use input halos, we can introduce a new flag to take another
        #soap fied: inclusive/exclusive_sphere_xxKpc
        self.catalogue_id=self.halo_properties.input_halos.halo_catalogue_index[index]
        self.centre=self.halo_properties.input_halos.halo_centre[index]
        self.half_mass_radius_gas=self.halo_properties.bound_subhalo.half_mass_radius_gas[index]

        return 

    def _most_bound_particles(self):

        bound_particles=self.halo_properties.input_halos.number_of_bound_particles
        most_bound_particles=np.max(bound_particles)
        most_bound_index=np.where(bound_particles==most_bound_particles)[0]
        
        #get halo info
        self._retrieve_halo(index=most_bound_index)
        
        #now retrieve the bound gas particles
        self._retrieve_bound_gas_particles()

        return 

    def _highest_gas_mass_halo(self):

        gas_mass_halos=self.halo_properties.bound_subhalo.gas_mass
        most_massive_halo=np.max(gas_mass_halos)
        most_gas_index=np.where(bound_particles==max_bound_particles)[0]
        
        #get halo info
        self._retrieve_halo(index=most_gas_index)
        
        #now retrieve the bound gas particles
        self._retrieve_bound_gas_particles()

        return

    