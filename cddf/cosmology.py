import numpy as np
import copy
import scipy.spatial as ss
from scipy.integrate import quad
from astropy.constants import c
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
import unyt as u
import astropy.units as au
from functools import cached_property
from swiftsimio.objects import cosmo_array, cosmo_factor



class cosmo_tools:
    
    def __init__(self,
                data_unpacker,
                cfg,
                update=None
                ):
            
        self.data_unpacker=data_unpacker
        self.cfg=cfg
        
        
        #now loading in constants
        constants = data_unpacker.cosmology
        
        update = {} if update is None else update
        #for colossus cosmology class
        if constants.name is not None:
            self.name=constants.name
        else: #for the swiftsimio cosmology class
            self.name=type(constants).__name__
        self.name= update.get("name",self.name)
        
        self.box_size=data_unpacker.box_size
        self.scale_factor=data_unpacker.scale_factor
        
        
        #Hubble constant

        #strip units if necessary, always in km/s/Mpc for fiducial and real cosmology
        if isinstance(constants.H0, au.quantity.Quantity):
            self.H0 = constants.H0.value
        else:
            self.H0 = constants.H0
        self.H0= update.get("H0",self.H0)
        self.h = self.H0 / 100.0

        #total matter
        self.Om0 = constants.Om0
        self.Om0= update.get("Om0",self.Om0)
        self.Omh2 = self.Om0 * self.h**2

        #baryons
        self.Ob0 = constants.Ob0
        self.Ob0= update.get("Ob0",self.Ob0)
        self.Obh2 = self.Ob0 * self.h**2

        #CDM
        self.Oc0= self.Om0-self.Ob0
        self.Oc0= update.get("Oc0",self.Oc0)
        self.Och2= self.Oc0 * self.h**2

        #dark energy
        self.Ode0 = constants.Ode0
        self.Ode0= update.get("Ode0",self.Ode0)
        if hasattr(constants, "w0"):
            self.w0=constants.w0
            self.w0= update.get("w0",self.w0)
        else:
            self.w0=-1.0
            self.w0= update.get("w0",self.w0)
        if hasattr(constants, "wa"):
            self.wa=constants.wa
            self.wa= update.get("wa",self.wa)
        else:
            self.wa=0.0
            self.wa= update.get("wa",self.wa)
        #check for w crossing -1
        if self.w0 != -1.0 or self.wa != 0.0:
            
            self.check_w_crossing()

        #CMB
        if isinstance(constants.Tcmb0, au.quantity.Quantity):
            self.Tcmb0 = constants.Tcmb0.value
        else:
            self.Tcmb0 = constants.Tcmb0
        self.Tcmb0= update.get("Tcmb0",self.Tcmb0)

        #neutrinos
        self.Neff = constants.Neff
        self.Neff= update.get("Neff",self.Neff)

        if hasattr(constants,"Onu0"):
            self.Onu0=constants.Onu0
            self.Onu0= update.get("Onu0",self.Onu0)
            self.Onuh2=self.Onu0*self.h**2
        if hasattr(constants,"nmassivenu"):
            self.nmassivenu=constants.nmassivenu
            self.nmassivenu= update.get("nmassivenu",self.nmassivenu)

        #radiation
        self.Omega_gamma=2.472e-5 * (self.Tcmb0 / 2.7255)**4 /(self.h)**2
        self.Or0 = self.Omega_gamma * (1.0 + 0.2271 * self.Neff)
        
        #curvature
        self.Ok0 = 1.0 - self.Om0 - self.Or0 - self.Ode0
        
        #constants
        self.c_km_s = c.to('km/s').value


        #position of the middle of the slice
        self.z_center = self.cfg.window.z_center
        #manually set the limits of the particles redshift. This is an educated guess. Better set to wide than to low
        self.z_min=self.cfg.window.z_range[0]
        self.z_max=self.cfg.window.z_range[1]

        




    # ----------------------------- Update method -----------------------------
    def update(self,params:dict =None, **kwargs):
        """
        Update one or more cosmological parameters and recalc everything.
        Example: cosmo.update(H0=70, Om0=0.31)
        """
        # store overrides
        new_update={}
        # combine dict and kwargs
        if params is not None:
            new_update.update(params)
        new_update.update(kwargs)
        #make a new init
        new_init = type(self)(
            box_size=self.box_size,
            constants=self.constants,
            redshift=self.redshift,
            redshift_bin_width=self.bin_width,
            update=new_update)
        return new_init

    def E(self, z):
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        
        # dynamical dark energy evolution
        if self.w0 != -1.0 or self.wa != 0.0:
            Odez = self.Ode0 * (1 + z)**(3 * (1 + self.w0 + self.wa)) * \
            np.exp(-3 * self.wa * z / (1 + z))
        #non-dynamical dark energy (w=-1)
        else:
            Odez = self.Ode0
        return np.sqrt(
            self.Om0 * (1 + z)**3 +
            self.Or0 * (1 + z)**4 +
            Odez +
            self.Ok0 * (1 + z)**2
        )





    def comoving_distance(self, z):
        """
        Compute comoving line-of-sight distance D_C(z) in Mpc.
        Works for scalar or array z.
        """
        z=np.atleast_1d(z)
        Dc_list = []

        for zi in z:
            integral, _ = quad(lambda zp: 1.0 / self.E(zp), 0.0, zi, epsrel=1e-6)
            Dc_i = (self.c_km_s / self.H0) * integral
            
            Dc_list.append(Dc_i)

        Dc_array = np.array(Dc_list) 
        
        Dc_cosmo = cosmo_array(
                            Dc_array,
                            u.Mpc,
                            comoving=True,
                            #set the scale factor at the value of the simulation snapshot
                            scale_factor=self.scale_factor, 
                            scale_exponent=1,
            )
        return Dc_cosmo if len(Dc_cosmo) > 1 else Dc_cosmo[0]
    
    
    def transverse_comoving_distance(self,z):
        """
        Compute transverse comoving distance D_M(z) in Mpc.
        """
        Dc = self.comoving_distance(z) # Mpc
        #open curvature case
        if self.Ok0 > 0:
            sqrt_Ok = np.sqrt(self.Ok0)
            Dm = (self.c_km_s / self.H0) / sqrt_Ok * \
            np.sinh(sqrt_Ok * Dc.value * self.H0 / self.c_km_s)
        #closed curvature case
        elif self.Ok0 < 0:
            sqrt_abs_Ok = np.sqrt(-self.Ok0)
            Dm = (self.c_km_s / self.H0) / sqrt_abs_Ok *\
             np.sin(sqrt_abs_Ok * Dc.value * self.H0 / self.c_km_s)
        #flat case
        else:
            Dm = Dc.value
        
        Dm_cosmo = cosmo_array(
                            Dm,
                            u.Mpc,
                            comoving=True,
                            scale_factor=self.scale_factor,
                            scale_exponent=1
                        )
        return Dm_cosmo if Dm_cosmo.size > 1 else Dm_cosmo[0]


    def luminosity_distance(self,z):
        """
        Compute luminosity distance D_L(z) in Mpc.
        """

        Dl = self.transverse_comoving_distance(z) * (1 + z)
        return Dl
    def angular_diameter_distance(self,z):
        """
        Compute angular diameter distance D_A(z) in Mpc.
        """
        Da = self.transverse_comoving_distance(z) / (1 + z)
        return Da

    #---function to transform D_c to z ---
    @cached_property
    def comoving_distance_to_redshift(self):
        #maybe increase resolution??
        z_grid = np.linspace(self.z_min, self.z_max, int(1e5)) 
        Dc_grid = np.array([self.comoving_distance(z).value for z in z_grid])  # Mpc
        # ensure monotonic
        assert np.all(np.diff(Dc_grid) > 0)
        inv_interp = PchipInterpolator(Dc_grid, z_grid, extrapolate=False)
        # call inv_interp(Dc_array) -> z_array (or raises for out-of-range)
        return inv_interp 

    def dX(self, column_length):


        d_center = self.comoving_distance(self.z_center)

        d_front = d_center - 0.5 * column_length.to_comoving()
        d_back  = d_center + 0.5 * column_length.to_comoving()

        delta_z = self.comoving_distance_to_redshift(d_back) - \
                self.comoving_distance_to_redshift(d_front)

        E_z_middle = self.E(self.z_center)

        dX = (delta_z * (1 + self.z_center)**2) / E_z_middle

        return dX

    #calculating effective cosmological functions within a certain redshift bin
    @staticmethod
    def effective_redshift(z):
        return np.mean(z)

    def effective_angular_diameter_distance(self,z):
        return self.angular_diameter_distance(self.effective_redshift(z))
    
    def effective_comoving_distance(self,z):
        return self.comoving_distance(self.effective_redshift(z))
    
    def effective_hubble_constant(self,z):
        return self.H0 * self.E(self.effective_redshift(z))*u.Unit('km/s/Mpc')