import numpy as np
import copy


class VisSim:
    def __init__(self, nants, vis_model_max=1000, gain_max=3, x_max=0.1, level="exact", verbose=False):
        """
        model_max: V^m_ij are drawn from a uniform distribution between 0.1*model_max and model_max.

        gain_max: g_bar values are drawn from a uniform distribution between 0.1*gain_max and gain_max.

        x_max: Linear gain offset values x are drawn from a uniform distribution between 
            0.1*x_max and x_max.

            CAN THESE VALUES BE NEGATIVE? The problem is that, when dealing with randomly 
            generated simulations, you can get a visibility value that is negative.

        I use uniform distributions so I can constrain the values, and make sure there are no negatives.
        """

        # check inputs 
        if nants <= 3 or vis_model_max <= 0 or gain_max <= 0 or x_max <= 0:
            raise ValueError("gls_sim: Invalid input value")

        proj = self.gain_proj_operator(nants, exclude_autos=True, dtype=np.float) # So we know the shape
        #print("proj shape:", proj.shape)

        # Setup model and gains and perturbations to generate observed visibilities

        V_model = vis_model_max*(0.1+np.random.random(size=proj.shape[0])*0.9)

        g_bar = gain_max*(0.1+np.random.random(size=proj.shape[1])*0.9)
        if verbose: print("g_bar", g_bar)

        x = x_max*(0.1+np.random.random(size=proj.shape[1])*0.9)  
        
        #x = np.array([0.1, -0.05, 0.04, 0.01])
        #V_model = np.ones_like(V_model)
        #g_bar = np.ones_like(x)
        

        self.nants = nants
        self.V_model = V_model
        self.g_bar = g_bar
        self.x = x
        self.proj = proj
        self.level = level
        self.verbose = verbose

        self.calculate_observed()

    def calculate_observed(self):
        # Generate the observed visibilities.
        V = np.empty(self.proj.shape[0])
        k = 0
        for i in range(self.nants):
            for j in range(i+1, self.nants):
                if self.level == "exact":
                    V[k] = self.g_bar[i]*self.g_bar[j]*self.V_model[k]*(1+self.x[i])*(1+self.x[j])
                else:
                    V[k] = self.g_bar[i]*self.g_bar[j]*self.V_model[k]*(1+self.x[i]+self.x[j])
                    #print(V[k], "=", self.g_bar[i],"*", self.g_bar[j], "*", self.V_model[k], "* (1+",self.x[i], "+", self.x[j],")")
                k += 1

        self.V = V

    def calculate_bl_gains(self):
        bl_gains = np.empty(self.proj.shape[0])
        k = 0
        for i in range(self.nants):
            for j in range(i+1, self.nants):
                if self.level == "exact":
                    bl_gains[k] = self.g_bar[i]*self.g_bar[j]*(1+self.x[i])*(1+self.x[j])
                else:
                    bl_gains[k] = self.g_bar[i]*self.g_bar[j]*(1+self.x[i]+self.x[j])
                k += 1

        return bl_gains


    def calculate_dy(self):
        g_bar = self.g_bar
        x = self.x
        V_model = self.V_model
        V_observed =  self.V
        dys = np.empty(self.proj.shape[0])
        k = 0
        for i in range(g_bar.size):
            for j in range(i+1, g_bar.size):

                # Calculate the difference in the observed visibility using the proposed
                # gain offsets g_vec.
                if self.level == "exact":
                    dy = g_bar[i] * g_bar[j]*(1 + x[i])*(1 + x[j]) * V_model[k] - V_observed[k]
                else: 
                    dy = g_bar[i] * g_bar[j]*(1 + x[i] + x[j]) * V_model[k] - V_observed[k]

                dys[k] = dy

                k += 1
    
        return dys

    def calculate_normalized_observed(self):
        V = np.empty(len(self.V))
        k = 0
        for i in range(self.nants):
            for j in range(i+1, self.nants):
                V[k] = self.V[k]/(self.g_bar[i]*self.g_bar[j]*self.V_model[k])-1
                k += 1
        return V

    def print(self):
        print("level", self.level)
        print("V_model", self.V_model)
        print("g_bar", self.g_bar)
        print("x", self.x)
        print("V", self.V)
        print("Bl gain", self.calculate_bl_gains())
        print("Normalized observed", self.calculate_normalized_observed())

    def gain_proj_operator(self, nants, exclude_autos=True, dtype=np.integer):
        """
        Projection operator for linearised gains, assuming the 
        linearisation g_i = \bar{g}_i(1 + x_i).

        This operator projects from a vector of gain perturbations 
        {x_i} to a visibility vector {V_ij}, and so will in general 
        be rectangular.

        Parameters
        ----------
        nants : int
            Number of antennas. A linear operator will be constructed 
            for all antenna pairs (with or without autos), ordered by 
            antenna i first, then j, e.g. {V_01, V_02, V_03, V_12, V_13, ...}

        exclude_autos : bool, optional
            Whether to exclude auto-baselines (i.e. where i=j) from the 
            projection. Default: True.

        dtype : dtype, optional
            Data type to use for the output array. Default: np.integer.

        Returns
        -------
        proj : array_like of int

        """

        # Exclude autos if requested
        auto_offset = 0
        if exclude_autos:
            auto_offset = 1

        # Calculate no. of visibilities
        nvis = nants * (nants - 1) // 2
        if not exclude_autos:
            nvis += nants

        # Construct operator by populating array with 1's where gain is present
        proj = np.zeros((nvis, nants), dtype=dtype)
        k = -1
        for i in range(nants):
            for j in range(i+auto_offset, nants):
                k += 1
                proj[k,i] = proj[k,j] = 1
        return proj
    
def perturb_gains(vis_sim, gain_perturb_percent):
    """
    gain_perturb_percent: Each g_bar value is increased or 
        reduced by this fraction. The choice of whether to increase/reduce is decided
        randomly. Don't use a Gaussian distribution for this because we want to limit
        and control the perturbation.
        If 0, then don't do the perturbation.
        
        Return a new object
    """
    
    new_vis_sim = copy.deepcopy(vis_sim)
    
    if gain_perturb_percent is not None:
        if gain_perturb_percent < 0:
            raise ValueError("gain_perturb_percent is < 0")
        if gain_perturb_percent > 0:
            # Perturb the orignal gains
            choices = np.random.random(size=len(new_vis_sim.g_bar))-0.5
            g_bar = np.where(choices <= 0, new_vis_sim.g_bar*(1-gain_perturb_percent/100.0), new_vis_sim.g_bar*(1+gain_perturb_percent/100.0))
            new_vis_sim.g_bar = g_bar
            
    return new_vis_sim

def perturb_vis(vis_sim, vis_perturb_percent):
    """
    vis_perturb_percent: Each V value is increased or
        reduced by this fraction. The choice of whether to increase/reduce is decided
        randomly. Don't use a Gaussian distribution for this because we want to limit
        and control the perturbation.
        If 0, then don't do the perturbation.

        Return a new object
    """

    new_vis_sim = copy.deepcopy(vis_sim)

    if vis_perturb_percent is not None:
        if vis_perturb_percent < 0:
            raise ValueError("vis_perturb_percent is < 0")
        if vis_perturb_percent > 0:
            # Perturb the orignal gains
            choices = np.random.random(size=len(new_vis_sim.V))-0.5
            V = np.where(choices <= 0, new_vis_sim.V*(1-gain_perturb_percent/100.0), new_vis_sim.V*(1+gain_perturb_percent/100.0))
            new_vis_sim.V = V

    return new_vis_sim
