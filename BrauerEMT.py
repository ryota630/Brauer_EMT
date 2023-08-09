import numpy as np

class Brauer_emt:
    def __init__(self):
        # ==========================
        # Constants
        # - - - - - - - - - - - - - - -
        self.mu = 1.25663706e-06  # vacuum permeability [m kg s-2 A-2]
        self.ep0 = 8.8542e-12     # vacuum permitivity [m-3 kg-1 s4 A2]
        self.c = 2.9979e+08       # speed of light [m/s]
        self.pi = np.pi           # pi
        self.n0 = 1.0             # refractive index of air
        # - - - - - - - - - - - - - - -
        # ==========================
        
        
    def EMT_twoD_SWS_to_neff(self,freq,n1,n2,f1,f2,p1,p2):
        '''
        * Def name
            EMT_twoD_SWS_to_neff
        * Description
            Calculate effective refractive index given area fraction
            Based on Brauer(1994): https://opg.optica.org/view_article.cfm?pdfKey=92654043-e0c1-4520-98b312f00908c33d_42237
        * input parameters
            - freq:   inpu frequency [Hz]
            - n1: refractive index of air, so 1.0
            - n2: refractive index of substrate
            - f1: area fraction along x axis
            - f2: area fraction along y axis
            - p1: pitch x
            - p2: pitch y

        * return
            - n_: 0th ordered effective refractive index
            - neff: 2nd ordered effective refractive index
        '''
        
        lamda = self.c/freq   # wavelength
        f = (f1+f2)/2.        # average fraction in x and y

        e1 = n1**2.*self.ep0  # refractive index --> permitivity (air) 
        e2 = n2**2.*self.ep0  # refractive index --> permitivity (substrate)

        ell_0 = (1.0 - f1)*e1+f1*e2    # Eq.1
        els_0 = 1./((1.-f2)/e1+f2/e2)  # Eq.2

        ell_2 = ell_0*(1.+(np.pi**2/3.)*(p1/lamda)**2.*f1**2*(1.-f1)**2.*((e2-e1)**2./(self.ep0*ell_0)))                    # Eq.3
        els_2 = els_0*(1.0+(np.pi)**2/3.0*(p2/lamda)**2*f2**2*(1.-f2)**2.*((e2-e1)**2.)*ell_0/self.ep0*(els_0/(e2*e1))**2.) # Eq.4
        
        e_2nd_up = (1.0 - f1)*e1 + f1*els_2           # Eq.6
        e_2nd_down = 1./((1.0 - f2)/e1 + f2/ell_2)    # Eq.7

        n_=(1-f**2)*n1+f**2*n2                        # Eq.5
        n__2nd_up = np.sqrt(e_2nd_up/(self.ep0))      # permitivity --> refractive index (up)
        n__2nd_down = np.sqrt(e_2nd_down/(self.ep0))  # permitivity --> refractive index (down)

        neff = 0.2*(n_+2.0*n__2nd_up+2.0*n__2nd_down) # Eq.8
        return n_*np.ones(len(neff)),neff
    
    def EMT_n_to_w(self,arr, n0, ns, vc, p):
        '''
        * Def name
            EMT_n_to_w
        * Description
            Find the geometrical width of structure from effective refractive index
            Based on Brauer(1994): https://opg.optica.org/view_article.cfm?pdfKey=92654043-e0c1-4520-98b312f00908c33d_42237
        * input parameters
            - arr: Refractive index (1D numpy array)
            - n0: refractive index of air, so 1.0
            - ns: refractive index of substrate
            - vc: center frequency [Hz]
            - p: pitch of the structure [m]

        * return
            - w: width [m]
        '''        

        area_frac_arr = np.linspace(0,1.0,10001)
        n_eff_arr = self.EMT_twoD_SWS_to_neff(vc,n0,ns,area_frac_arr,area_frac_arr,p,p)
        
        w = np.zeros(len(arr))
        for i in range(0,len(w)):
            w[i] = p*area_frac_arr[np.argmin(abs(arr[i] - n_eff_arr[1]))]
        return w
    
    def Design_index_one_to_three_layer(self,ns):
        # Single_ARC
        n1_single = np.sqrt(self.n0*ns)
        
        # Two-layer ARC
        n1_two = (self.n0**2 * ns)**(0.25)
        n2_two = (self.n0 * ns**3)**(0.25)
        
        #Three-layer ARC
        n1_three = (self.n0**3 * ns)**(0.25)
        n2_three = (self.n0**2 * ns**2)**(0.25)
        n3_three = (self.n0**1 * ns**3)**(0.25)
        
        n_single = np.array([n1_single])
        n_two = np.array([n1_two,n2_two])
        n_three = np.array([n1_three,n2_three,n3_three])
        
        return n_single, n_two, n_three
