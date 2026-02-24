import numpy as np
import copy
import scipy.spatial as ss
from scipy.integrate import quad
from astropy.constants import c
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
import unyt as u
import astropy.units as au



class cosmo_tools:
    
    def __init__(self,
    box_size,
    constants,
    redshift,
    redshift_bin_width,
    update=None
    ):
        self.constants = constants
        update = {} if update is None else update
        #for colossus cosmology class
        if constants.name is not None:
            self.name=constants.name
        else: #for the swiftsimio cosmology class
            self.name=type(constants).__name__
        self.name= update.get("name",self.name)
        self.box_size=box_size

        
        
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

        


        #save all the importan parameters here
        self.redshift=redshift
        
        self.bao_distance=self._bao_sound_horizon()
    
        
        

        #set outer and inner edges of the redshift bin
        self.bin_width=redshift_bin_width
        self._edges_bin()

        #---function to transform D_c to z ---
        #call cosmology.cosmo_tools.comoving_distance_to_redshift(redshift)
        self.comoving_distance_to_redshift=self._comoving_distance_to_redshift()

        #complete sphere and the maximum angle, handy to store here:
        self._observer_position()


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
    

    def check_w_crossing(self):
        w_today = self.w0
        w_early = self.w0 + self.wa
        
        if (w_today + 1) * (w_early + 1) > 0:
            self.w_crossing=False
        else:
            self.w_crossing=True
    
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

    @staticmethod
    def redshift_with_error(z):
        #randomly distributes point in the z (radial) axis
        #we ignore systematic errors for now
        sigma_z=0.0005*(1+z)
        random_z=np.random.normal(z,sigma_z)
        return random_z

    
        
    def _edges_bin(self):
        half_binwidth=self.bin_width/2
        self.min_redshift=self.redshift-half_binwidth
        self.max_redshift=self.redshift+half_binwidth
        self.outer_edge_bin=self.comoving_distance(self.max_redshift)
        self.inner_edge_bin=self.comoving_distance(self.min_redshift)
        self.center_bin=self.comoving_distance(self.redshift)
        self.delta_dr=self.outer_edge_bin-self.inner_edge_bin

    def _observer_position(self):
        if self.outer_edge_bin < 0.5 * self.box_size:
            self.complete_sphere=True
            self.max_angle=np.pi
        else:
            self.complete_sphere=False
            self.max_angle=np.arcsin(self.box_size / (2 *self.outer_edge_bin))*u.rad

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

        Dc_array = np.array(Dc_list) * u.Mpc
        return Dc_array if len(Dc_array) > 1 else Dc_array[0]
    
    def transverse_comoving_distance(self,z):
        """
        Compute transverse comoving distance D_M(z) in Mpc.
        """
        Dc = self.comoving_distance(z) # Mpc
        #open curvature case
        if self.Ok0 > 0:
            sqrt_Ok = np.sqrt(self.Ok0)
            Dm = (self.c_km_s / self.H0) / sqrt_Ok * \
            np.sinh(sqrt_Ok * Dc.value * self.H0 / self.c_km_s)* u.Mpc
        #closed curvature case
        elif self.Ok0 < 0:
            sqrt_abs_Ok = np.sqrt(-self.Ok0)
            Dm = (self.c_km_s / self.H0) / sqrt_abs_Ok *\
             np.sin(sqrt_abs_Ok * Dc.value * self.H0 / self.c_km_s) * u.Mpc
        #flat case
        else:
            Dm = Dc.value * u.Mpc
        
        return Dm


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

    def _bao_sound_horizon(self):
        """
        Compute BAO comoving sound horizon r_d at the drag epoch.
        Based on Eisenstein & Hu (1998).
        """
        # --- Drag epoch redshift ---
        b1 = 0.313 * self.Omh2**(-0.419) * (1 + 0.607 * self.Omh2**0.674)
        b2 = 0.238 * self.Omh2**0.223
        z_drag = 1291 * self.Omh2**0.251 / (1 + 0.659 * self.Omh2**0.828) *\
         (1 + b1 * self.Obh2**b2)

        # --- Sound horizon integral ---
        def R_of_z(zp):
            return (3.0 * self.Ob0) / (4.0 * self.Omega_gamma) / (1.0 + zp)

        def c_s(zp):  
            return self.c_km_s / np.sqrt(3.0 * (1.0 + R_of_z(zp)))

        def integrand(zp):
            return c_s(zp) / (self.H0 *self.E(zp))

        r_d, _ = quad(integrand, z_drag, 1e7, epsrel=1e-6, limit=200)
        return r_d*u.Mpc

    def _comoving_distance_to_redshift(self):
        #builds a mapping of z <--> D_c
        #maybe increase resolution??
        z_grid = np.linspace(self.min_redshift, self.max_redshift, int(1e5)) 
        Dc_grid = np.array([self.comoving_distance(z).value for z in z_grid])  # Mpc
        # ensure monotonic
        assert np.all(np.diff(Dc_grid) > 0)
        inv_interp = PchipInterpolator(Dc_grid, z_grid, extrapolate=False)
        # call inv_interp(Dc_array) -> z_array (or raises for out-of-range)
        return inv_interp 



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