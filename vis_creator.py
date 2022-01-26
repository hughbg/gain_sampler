import numpy as np
import copy
import hickle as hkl
from pyuvdata import UVData, UVCal
from calcs import calc_visibility, split_re_im, unsplit_re_im, BlockMatrix

class VisSim:
    """
    Simulate an observation. Create random model, gains, gain offsets (x).
    Then generate initial observed visibilities as:
    
        V_obs[k] = V_model[k] g[i] conj(g[j]) (1 + x[i] + conj(x[j]))   ["approx"]
        
        OR 
        
        V_obs[k] = V_model[k] g[i] conj(g[j]) (1 + x[i]) (1 + conj(x[j]))   ["exact"]
        
    depending on the "level" of exactness to be applied (approx/exact). GLS can only use
    the first form.
        
    Simulated values are stored in the object. If any of them are perturbed from their 
    initial values after initialization, the equation above will not hold.
    
    The RHS of the above equations can always be calculated by calling
    get_simulated_visibilities().
    """
    
 
    def __init__(self, nant, ntime=1, nfreq=1, vis_model_sigma=1000, gain_sigma=3, x_sigma=0.1, obs_variance=10+10j, 
                 level="approx", redundant=False):
        """
        Parameters
        ----------
        nant: integer
            Number of antennas, which then defines the number of baselines. 
            Autocorrelation baselines are not generated.
            
        vis_model_sigma: float, optional
            Visibility real/imag values will be drawn from a Gaussian distribution
            with this standard deviation. 
            
        gain_sigma: float, optional
            Similar to vis_model_sigma but for gains. 
            
        x_sigma: float, optional
            Similar to vis_model_sigma but for gain offsets (x). 
        
        obs_variance: float, optional
            Variance on the observed visibilities V_obs.
            
        level: str
            Specifies whether to use the gain offsets exactly or approximately.
            Options: [ "approx", "exact" ]
            "approx" means calculate baseline gains using gi*conj(gj)*(1+xi+conj(xj). 
            "exact" means calculate baseline gains using gi*conj(gj)*(1+xi)*(1+conj(xj)).
            
        redundant: bool
            If True then V_model is made the same for all baselines.
         
        """

        # check inputs 
        if nant <= 2 or vis_model_sigma < 0 or gain_sigma < 0 or x_sigma < 0 \
                or level not in [ "approx", "exact" ]:
            raise ValueError("VisSim: Invalid input value")
        if ntime < 1 or nfreq < 1:
            raise ValueError("ntime and nfreq must be > 0")
            
        if x_sigma > 0.1:
            print("WARNING: x_sigma is above the recommend (0.1)")

        nvis = nant * (nant - 1) // 2
        
        # Setup model and gains and x to generate observed visibilities
        V_model = np.empty((ntime, nfreq, nvis), dtype=np.complex64)
        
        for time in range(ntime):
            for freq in range(nfreq):
                re = np.random.normal(scale=vis_model_sigma, size=nvis)
                im = np.random.normal(scale=vis_model_sigma, size=nvis)
                V_model[time, freq] = re+1j*im
                if redundant:
                    V_model[time, freq, :] = V_model[time, freq, 0]

        g_bar = np.empty((ntime, nfreq, nant), dtype=np.complex64)
        for time in range(ntime):
            for freq in range(nfreq):              
                re = np.random.normal(scale=gain_sigma, size=nant)
                im = np.random.normal(scale=gain_sigma, size=nant)
                g_bar[time, freq] = re+1j*im

        x = np.empty((ntime, nfreq, nant), dtype=np.complex64)
        for time in range(ntime):
            for freq in range(nfreq):              
                re = np.random.normal(scale=x_sigma, size=nant)
                im = np.random.normal(scale=x_sigma, size=nant)
                x[time, freq] = re+1j*im

        self.nant = nant
        self.nvis = nvis
        self.ntime = ntime
        self.nfreq = nfreq
        self.V_model = self.V_model_orig = V_model
        self.g_bar = self.g_bar_orig = g_bar
        self.x = self.x_orig = x
        self.level = level
        self.obs_variance = np.empty((ntime, nfreq, nvis), dtype=type(V_model[0, 0, 0]))
        for time in range(ntime):
            for freq in range(nfreq): 
                self.obs_variance[time][freq] = np.full(nvis, obs_variance) 
        if redundant: self.redundant_groups = None
        else:
            self.redundant_groups = [ [bl] for bl in range(nvis) ]

        self.model_projection = self.get_model_projection()
        self.V_obs = self.get_simulated_visibilities()


    def get_simulated_visibilities(self, show_working=False):
        """
        Calculate V_model*gains*x for all baselines. 
        
        Returns
        -------
        
        array_like
            The new visibilities, indexed by baseline.
        """
        
        V_obs = np.empty((self.ntime, self.nfreq, self.nvis), dtype=type(self.V_model[0, 0, 0]))
        V_model = self.project_model()

        for time in range(self.ntime):
            for freq in range(self.nfreq):
                V = np.empty(self.nvis, dtype=type(self.V_model))    
                k = 0
                for i in range(self.nant):
                    for j in range(i+1, self.nant):
                        V[k] = calc_visibility(self.level, self.g_bar[time, freq, i], self.g_bar[time, freq, j], 
                                               V_model[time, freq, k], self.x[time, freq, i], self.x[time, freq, j])
                        if show_working:
                            if self.level == "approx":
                                print(self.V_obs[k], "=?", "[", V[time, freq, k], "]", V_model[time, freq, k], "x", self.g_bar[time, freq, i], "x", np.conj(self.g_bar[time, freq, j]), 
                                    "(1 +", self.x[time, freq, i], "+", np.conj(self.x[time, freq, j]), ")")
                            else:
                                print(self.V_obs[time, freq, k], "=?", "[", V[time, freq, k], "]", V_model[time, freq, k], "x", self.g_bar[time, freq, i], "x", np.conj(self.g_bar[time, freq, j]), 
                                    "( 1 +", self.x[time, freq, i], ") x ( 1 +", np.conj(self.x[time, freq, j]), "))")

                        k += 1
                V_obs[time, freq] = V
                
        return V_obs
    
    def get_baseline_gains(self):
        """
        Calculate the gains applied to the model to simulate visibilities. 
        This is the factor g[i]*conj(g[j])*(1+x[i]+conj(x[j])) or
        g[i]*conj(g[j])*(1+x[i])*(1+conj(x[j]) depending on self.level.
        
        Returns
        -------
        
        array_like
            The gains, indexed by baseline.
        """

        V_sim = self.get_simulated_visibilities()    
        
        return V_sim/self.project_model()

    def get_antenna_gains(self):
        """
        Calculate the antenna gains by incorporating the gain offsets for all 
        antennas.
        
            gain[k] = g[k] (1 + x[k])
        
        Returns
        -------
        
        array_like
            The gains, indexed by antenna.
        """
        
        return self.g_bar*(1+self.x)
    

    def get_reduced_observed(self):
        """
        Reduced visibilities (Vr) are the observed visibilities minus
        the model times gains (not including gain offsets). Calculate:
        
            Vr[k] = V_obs[k] - V_model[k] g[i] conj(g[j])
            
        These are used in the GLS to solve the equation Vr = P x where
        P is the projection operator and x are the gain offsets.
        
        Returns
        -------
        
        array_like
            The reduced visibilities, indexed by baseline.
        """
        
        V_model = self.project_model()      # time, freq, nvis
        V = np.empty((self.ntime, self.nfreq, self.nvis), dtype=type(self.V_obs[0, 0, 0]))
        for time in range(self.ntime):
            for freq in range(self.nfreq):
                k = 0
                for i in range(self.nant):
                    for j in range(i+1, self.nant):     
                        V[time, freq, k] = self.V_obs[time, freq, k]-V_model[time, freq, k] \
                                *self.g_bar[time, freq, i]*np.conj(self.g_bar[time, freq, j])
                        k += 1
        return V
    
    def get_reduced_observed1(self):
        """
        Reduced visibilities (Vr) are the observed visibilities minus
        the model times gains (not including gain offsets). Calculate:
        
            Vr[k] = V_obs[k] - V_model[k] g[i] conj(g[j])
            
        These are used in the GLS to solve the equation Vr = P x where
        P is the projection operator and x are the gain offsets.
        
        Returns
        -------
        
        array_like
            The reduced visibilities, indexed by baseline.
        """
        
        V_model = self.project_model()      # time, freq, nvis
        V = np.empty((self.ntime, self.nfreq, self.nvis), dtype=type(self.V_obs[0, 0, 0]))

        for time in range(self.ntime):
            for freq in range(self.nfreq):
                k = 0
                for i in range(self.nant):
                    for j in range(i+1, self.nant):
                        V[time, freq, k] = self.V_obs[time, freq, k]/V_model[time, freq, k] \
                                /(self.g_bar[time, freq, i]*np.conj(self.g_bar[time, freq, j]))-1
                        k += 1
        return V
 

    def get_calibrated_visibilities(self):
        """
        Divide the observed visibilities by the baseline gains.
        """
        
        return self.V_obs/self.get_baseline_gains()
    

    def get_unnormalized_likelihood(self, over_all=False, unity_N=False):
        """
        Calculate the RMS of the real/imag values from get_diff_obs_sim()
        """

        dy = self.V_obs-self.get_simulated_visibilities()
           
        if over_all:
            dy_all = split_re_im(np.ravel(dy))
            N_inv_all = BlockMatrix()
            for time in range(self.ntime):
                for freq in range(self.nfreq):
                    if unity_N:
                        N_inv_all.add(np.diag(np.full(self.obs_variance[time][freq].size*2, 1)))
                    else: 
                        N_inv_all.add(np.linalg.inv(np.diag(split_re_im(self.obs_variance[time][freq]))))
            N_inv_all = N_inv_all.assemble()
            
            return np.exp(-0.5*np.dot(dy_all, np.dot(N_inv_all, dy_all)))
        
        else:
            likelihoods = np.zeros((self.ntime, self.nfreq))
            for time in range(self.ntime):
                for freq in range(self.nfreq):
                    dy_tf = split_re_im(dy[time, freq])
                    if unity_N:
                        N_inv = np.diag(np.full(self.obs_variance[time][freq].size*2, 1))
                    else:
                        N_inv = np.linalg.inv(np.diag(split_re_im(self.obs_variance[time][freq])))
                    likelihoods[time, freq] = np.exp(-0.5*np.dot(dy_tf, np.dot(N_inv, dy_tf)))
            return likelihoods
                    
    
    def get_model_projection(self):
        # Will be the same for all times/freqs (assuming redundant groups are all the same)
        mapping = self.map_to_redundant_group()
        P = np.zeros((self.nvis*2, len(self.redundant_groups)*2))
        for i in range(P.shape[0]//2):      # This will scan baselines
            P[i*2][mapping[i]*2] = 1
            P[i*2+1][mapping[i]*2+1] = 1  
            
        return P
    
    def project_model(self):
        V_model = np.empty((self.ntime, self.nfreq, self.nvis), dtype=np.complex64)
        for time in range(self.ntime):
            for freq in range(self.nfreq):
                V = split_re_im(self.V_model[time, freq])
                V_model[time, freq] = unsplit_re_im(np.dot(self.model_projection, V))
        return V_model


    def get_chi2(self):
        """
        The chi2 of the fit between V_obs and visibilities generated by V_model*gains*x
        """
        return np.sum(np.abs(self.V_obs-self.get_simulated_visibilities())**2/self.obs_variance)
    
    def get_best_antenna(self):
        diffs = np.abs(self.get_simulated_visibilities()-self.V_obs)
        return np.argmin(diffs)
    
    def map_to_redundant_group(self):
        
        mapping = []
        red_index = 0
        for red in self.redundant_groups:
            for bl in red:
                mapping.append((bl, red_index))
            red_index += 1
        mapping.sort()

        return [ mapped[1] for mapped in mapping ]

    def list_redundant_groups(self):
        mapping = self.map_to_redundant_group()
        for red in v.redundant_groups:
            for bl in red:
                print(mapping[bl], bl, v.V_model[bl])
                                
    def print(self):
        print("V_model", self.V_model)
        print("g_bar", self.g_bar)
        print("x", self.x)
        print("V obs", self.V_obs)
        print("Bl gain", self.get_baseline_gains())
        print("Normalized observed", self.get_reduced_observed())
        print("level", self.level)
        print("Chi2", self.get_chi2())
        print("Quality", self.get_quality())


class VisCal(VisSim):
    """
    Data from a non-redundant-pipeline simulation, including calibration.
    """
    # Warning: hickle file needed
    # TODO: make these steps separate functions
    def __init__(self, file_root, time_range=None, freq_range=None, remove_redundancy=False, degeneracy_fix=False, level="approx"):
        """
        file_root: str
            The file path referring to a non-redundant-pipeline simulation "case". 
            Use that path to select files from the simulation based on their extension.
        """
        
        
        # Get V_obs from last file generated by analyse_sims.py 
        
        fname = file_root+"_g.uvh5"
        print("Get V_obs from", fname)
        uvdata = UVData()
        uvdata.read_uvh5(fname)
        assert uvdata.Nants_data == uvdata.Nants_telescope, \
                "Not all antennas have data"
        nant = uvdata.Nants_data
        # Map antenna numbers to the vis index
        bl_to_ants = []
        for i in range(nant):
            for j in range(i+1, nant):
                bl_to_ants.append((i, j))
        assert uvdata.Nbls-nant == len(bl_to_ants), \
                "Data does not contain expected number of baselines"
        nvis = len(bl_to_ants)
        
        if time_range is None:
            time_range = ( 0, uvdata.Ntimes )
        if freq_range is None:
            freq_range = ( 0, uvdata.Nfreqs )
       
        ntime = time_range[1]-time_range[0]
        nfreq = freq_range[1]-freq_range[0]

        # Load V_obs
        V = np.zeros((ntime, nfreq, nvis), dtype=type(uvdata.get_data(bl_to_ants[0][0], bl_to_ants[0][1], "XX")[0, 0]))
        for i, bl in enumerate(bl_to_ants):
            V[:, :, i] = uvdata.get_data(bl[0], bl[1], "XX")[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]

        # Get things out of calibration: model, redundant groups, weights

        fname = file_root+"_g_cal_dict.npz"
        print("Get model from", fname)
        cal = hkl.load(fname)

        baseline_groups = cal["all_reds"]
        redundant_groups = [ [] for bg in baseline_groups ]
        assert len(cal["v_omnical"].keys()) == len(baseline_groups), \
                "Number of baseline groups: "+str(len(baseline_groups))+ \
                " Number of model groups: "+str(len(cal["v_omnical"].keys()))
        V_model = np.zeros((ntime, nfreq, nvis), dtype=type(V[0, 0, 0]))
        for key in cal["v_omnical"].keys():
            baseline = (key[0], key[1])
            
            # Find the baselines in this group. First item will be "ants"
            group_index = -1
            for i in range(len(baseline_groups)):
                if (baseline_groups[i][0][0], baseline_groups[i][0][1]) == baseline:
                    group_index = i
                    break
            if group_index == -1:
                raise RuntimeError("Couldn't find redundant group for model "+str(key))
                    
            for bl in baseline_groups[group_index]:
                try:
                    i = bl_to_ants.index((bl[0], bl[1]))
                    V_model[:, :, i] = cal["v_omnical"][key][time_range[0]:time_range[1], freq_range[0]:freq_range[1]]
                except:
                    i = bl_to_ants.index((bl[1], bl[0]))
                    V_model[:, :, i] = np.conj(cal["v_omnical"][key])[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]

                redundant_groups[group_index].append(i)

        assert np.min(np.abs(V_model)) > 0, "Model has missing values"

        # Check that the models in the redundant groups are all equal
        for red in redundant_groups:
            for bl in red[1:]:
                for time in range(ntime):
                    for freq in range(nfreq):
                        assert V_model[time, freq, bl] == V_model[time, freq, red[0]]
                
        # Now we are going to fiddle with the redundancy and compact the 
        # list of models if there is redundancy

        if remove_redundancy: 
            redundant_groups = [ [bl] for bl in range(nvis) ]
        else:
            new_V_model = np.empty((ntime, nfreq, len(redundant_groups)), dtype=np.complex64)
            for time in range(ntime):
                for freq in range(nfreq):
                    for i in range(len(redundant_groups)):
                        new_V_model[time, freq, i] = V_model[time, freq, redundant_groups[i][0]]
            V_model = new_V_model
            

        """
        # Get the weights used by calibration
        num_weights = len(list(cal["omni_meta"]["data_wgts"]))
        assert num_weights == V.size, "Not enough weights for baselines "+str(num_weights)+" "+str(V.size)
        weights = np.zeros(V.size)
        for key in cal["omni_meta"]["data_wgts"].keys():
            try:
                i = bl_to_ants.index((key[0], key[1]))
            except:
                i = bl_to_ants.index((key[1], key[0]))
            weights[i] = cal["omni_meta"]["data_wgts"][key][time][freq]
        """
        
        # Get the noise
        fname = file_root+"_nn.uvh5"
        print("Get noise from", fname)
        uvdata = UVData()
        uvdata.read_uvh5(fname)
        noise = np.zeros((ntime, nfreq, nvis), dtype=type(V[0, 0, 0]))
        for i, bl in enumerate(bl_to_ants):
            noise[:, :, i] = uvdata.get_data(bl[0], bl[1], "XX")[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]

        # Get the gains generated from calibration
        g_bar = np.zeros((ntime, nfreq, nant), dtype=type(V[0]))
        if degeneracy_fix:
            fname = file_root+"_g_new.calfits"
            print("Get gains from", fname)
            uvc = UVCal()
            uvc.read_calfits(fname)
            for i in range(nant):
                # The data for each gain is in shape (nfreq, ntime)
                # Yes it's the other way round to below
                g_bar[:, :, i] = uvc.get_gains(i).T[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]
            self.gain_degeneracies_fixed = True
            
        else:
            # Get results from cal (g_omnical).
            assert len(cal["g_omnical"].keys()) == nant
            for key in cal["g_omnical"]:
                # Key is of form (0, 'Jee'). The data for each key is in shape (ntime, nfreq)
                ant = key[0]
                g_bar[:, :, ant] = cal["g_omnical"][key][time_range[0]:time_range[1], freq_range[0]:freq_range[1]]

            
        self.file_root = file_root
        self.ntime = ntime
        self.nfreq = nfreq

        self.level = level
        self.nant = nant
        self.nvis = nvis
        self.V_obs = V
        self.V_model = V_model
        self.g_bar = g_bar
        self.x = np.zeros_like(g_bar, dtype=type(self.g_bar[0,0,0]))   # Yes teh dtype is needed, weird things with complex128
        # The noise is standard deviation of re/im parts for all visibilities.
        # Convert these values into variances.
        self.obs_variance = noise
        self.redundant_groups = redundant_groups
        self.bl_to_ants = bl_to_ants
        self.model_projection = self.get_model_projection()
        

class VisTrue(VisSim):
    """
    Data from a non-redundant-pipeline simulation, not including calibration.
    """
    def __init__(self, file_root, time_range=None, freq_range=None, level="approx"):
        # Get model

        fname = file_root+".uvh5"
        print("Get true model from", fname)
        uvdata = UVData()
        uvdata.read_uvh5(fname)
        assert uvdata.Nants_data == uvdata.Nants_telescope, \
                "Not all antennas have data"
        nant = uvdata.Nants_data
        
        # Map antenna numbers to the vis index
        bl_to_ants = []
        for i in range(nant):
            for j in range(i+1, nant):
                bl_to_ants.append((i, j))
        assert uvdata.Nbls-nant == len(bl_to_ants), \
                "Data does not contain expected number of baselines"
        nvis = len(bl_to_ants)
        
        if time_range is None:
            time_range = ( 0, uvdata.Ntimes )
        if freq_range is None:
            freq_range = ( 0, uvdata.Nfreqs )
       
        ntime = time_range[1]-time_range[0]
        nfreq = freq_range[1]-freq_range[0]
        
        # Load model
        V_model = np.zeros((ntime, nfreq, nvis), dtype=type(uvdata.get_data(bl_to_ants[0][0], bl_to_ants[0][1], "XX")[0, 0]))
        for i, bl in enumerate(bl_to_ants):
            V_model[:, :, i] = uvdata.get_data(bl[0], bl[1], "XX")[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]
            
        # Wipe out the redundant groups because the models are not the same in the redundant groups
        redundant_groups = [ [bl] for bl in range(nvis) ]

        # Get true gains

        fname = file_root+".calfits"
        print("Get true gains from", fname)
        uvc = UVCal()
        uvc.read_calfits(fname)
        g_bar = np.zeros((ntime, nfreq, nant), dtype=type(V_model[0, 0, 0]))
        for i in range(nant):
            g_bar[:, :, i] = uvc.get_gains(i).T[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]
    

        # Get V_obs

        fname = file_root+"_g.uvh5"
        print("Get V_obs from", fname)
        uvdata = UVData()
        uvdata.read_uvh5(fname)
        assert uvdata.Nants_data == uvdata.Nants_telescope, \
                "Not all antennas have data"
        assert uvdata.Nbls-nant == nvis, \
                "Data does not contain expected number of baselines"

        # Load V_obs
        V = np.zeros((ntime, nfreq, nvis), dtype=type(V_model[0, 0, 0]))
        for i, bl in enumerate(bl_to_ants):
            V[:, :, i] = uvdata.get_data(bl[0], bl[1], "XX")[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]
            

        self.level = level
        self.nant = nant
        self.nvis = nvis
        self.ntime = ntime
        self.nfreq = nfreq
        self.nvis = nvis
        self.g_bar = g_bar
        self.V_model = V_model
        self.x = np.zeros_like(g_bar)
        self.V_obs = V
        self.obs_variance = np.empty((ntime, nfreq, nvis), dtype=type(V_model[0, 0, 0]))
        for time in range(ntime):
            for freq in range(nfreq): 
                self.obs_variance[time][freq] = np.full(nvis, 1+1j) 
        self.redundant_groups = redundant_groups

        self.bl_to_ants = bl_to_ants
        self.model_projection = self.get_model_projection()
        




def perturb_gains(vis, gain_perturb_percent):
    """
    
    gain_perturb_percent: Each g_bar value is increased or 
        reduced by this fraction. The choice of whether to increase/reduce is decided
        randomly. Don't use a Gaussian distribution for this because we want to limit
        and control the perturbation.
        If 0, then don't do the perturbation.
        
        Return a new object
    """
    

    new_vis = copy.deepcopy(vis)
    
    if gain_perturb_percent is not None:
        if gain_perturb_percent < 0:
            raise ValueError("gain_perturb_percent is < 0")
        if gain_perturb_percent > 0:
            # Perturb the orignal gains
            choices = np.random.random(size=len(new_vis.g_bar))-0.5
            g_bar = np.where(choices <= 0, new_vis.g_bar*(1-gain_perturb_percent/100.0), new_vis.g_bar*(1+gain_perturb_percent/100.0))
            new_vis.g_bar = g_bar
            
    return new_vis

def select_redundant_group(vis, index):
    assert vis.redundant_groups is not None, "No redundant groups"
    assert index < len(vis.redundant_groups), "No redundant group "+str(index)
    
    # Find out how many ants are involved in this group, and setup
    # based on that, but the baselines may be a lot fewer than the
    # ants allow.
    
    group_ants = [ vis.bl_to_ants[i] for i in range(len(vis.bl_to_ants)) if i in vis.redundant_groups[index] ]

    ants = [ tup[0] for tup in group_ants ] + [ tup[1] for tup in group_ants ]
    ants = sorted(list(set(ants)))

    # The ants are now going to be renumbered, because some might be missing.
    # The new numbers are just the index in the list "ants".
    # nants are not goint to match the number of baselines using the usual
    # formula.
        
    new_vis = VisSim(len(ants), redundant=True)
    new_vis.nvis = len(vis.redundant_groups[index])
    new_vis.V_obs = np.zeros(new_vis.nvis, dtype=type(vis.V_obs[0]))
    new_vis.V_model = np.zeros(new_vis.nvis,  dtype=type(vis.V_model[0]))

    j = 0
    new_vis.bl_to_ants = []
    for bl in vis.redundant_groups[index]:
        new_vis.V_model[j] = vis.V_model[bl]
        new_vis.V_obs[j] = vis.V_obs[bl]
        new_vis.obs_variance[j] = vis.obs_variance[bl]      # Baseline variances
        
        ant_left = ants.index(vis.bl_to_ants[bl][0])
        ant_right = ants.index(vis.bl_to_ants[bl][1])
        new_vis.bl_to_ants.append((ant_left, ant_right))
                       
        j += 1
        
    for j, ant in enumerate(ants):
        new_vis.g_bar[j] = vis.g_bar[ant]
        new_vis.x[j] = vis.x[ant]
        
    new_vis.redundant_groups = [ list(range(new_vis.nvis)) ]
    

    return new_vis

def select_no_redundancy(vis):
    baselines = [ group[0] for group in vis.redundant_groups ]
    
    # Find out how many ants are involved in this group, and setup
    # based on that, but the baselines may be a lot fewer than the
    # ants allow.
    
    group_ants = [ vis.bl_to_ants[i] for i in range(len(vis.bl_to_ants)) if i in baselines ]

    ants = [ tup[0] for tup in group_ants ] + [ tup[1] for tup in group_ants ]
    ants = sorted(list(set(ants)))

    # The ants are now going to be renumbered, because some might be missing.
    # The new numbers are just the index in the list "ants".
        
    new_vis = VisSim(len(ants), redundant=True)
    new_vis.nvis = len(baselines)
    new_vis.V_obs = np.zeros(new_vis.nvis, dtype=type(vis.V_obs[0]))
    new_vis.V_model = np.zeros(new_vis.nvis,  dtype=type(vis.V_model[0]))

    j = 0
    new_vis.bl_to_ants = []
    for bl in baselines:
        new_vis.V_model[j] = new_vis.initial_vals.V_model[j] = vis.V_model[bl]
        new_vis.V_obs[j] = new_vis.initial_vals.V_obs[j] = vis.V_obs[bl]
        new_vis.obs_variance[j] = vis.obs_variance[bl]      # Baseline variances
        
        ant_left = ants.index(vis.bl_to_ants[bl][0])
        ant_right = ants.index(vis.bl_to_ants[bl][1])
        new_vis.bl_to_ants.append((ant_left, ant_right))
                       
        j += 1
        
    for j, ant in enumerate(ants):
        new_vis.g_bar[j] = vis.g_bar[ant]
        new_vis.x[j] = vis.x[ant]
        
    new_vis.redundant_groups = None

    return new_vis


def perturb_vis(vis, vis_perturb_percent):
    """
    vis_perturb_percent: Each V value is increased or
        reduced by this fraction. The choice of whether to increase/reduce is decided
        randomly. Don't use a Gaussian distribution for this because we want to limit
        and control the perturbation.
        If 0, then don't do the perturbation.

        Return a new object
    """

    new_vis = copy.deepcopy(vis)

    if vis_perturb_percent is not None:
        if vis_perturb_percent < 0:
            raise ValueError("vis_perturb_percent is < 0")
        if vis_perturb_percent > 0:
            # Perturb the orignal vis
            #vals = split_re_im(new_vis.V_obs)
            #choices = np.random.random(size=len(vals))-0.5
            #new_vis.V_obs = unsplit_re_im(np.where(choices <= 0, vals*(1-vis_perturb_percent/100.0), 
            #                                       vals*(1+vis_perturb_percent/100.0)))
            for i in range(new_vis.V_obs.size):
                new_vis.V_obs[i] = complex(new_vis.V_obs[i].real+np.random.normal(scale=100), new_vis.V_obs[i].imag+np.random.normal(scale=100))

    return new_vis

def group_by_x(a):
    # Reshape so the dimensions are nant, ntime, nfreq
    return np.moveaxis(a, 2, 0)

def operator_reorder_x_after_fft(nt, nf,nx ): 
    # For a single time
    
    # Make blocks by frequency and combine them

    op = np.zeros((nx*2*nf, nx*2*nf)) 
    k = 0
    for ix in range(nx):            
        for re_im in range(2):
            for i in range(nf):
                op[2*nx*i+ix*2+re_im, k] = 1
                k += 1
    
    bm = BlockMatrix()
    for t in range(nt):
        bm.add(op)
    
    return bm.assemble()

                         
def multi_fft_operator(nt, nf, nx):
    op = fourier_operator(nf)
    bm = BlockMatrix()
    for i in range(nt):
        for j in range(nx):
            for k in range(2):        # real/imag
                bm.add(op)
                
    return bm.assemble()

if __name__ == "__main__":
    from gls import generate_proj
    from calcs import fourier_operator
    
    # Check d = A x
    v = VisSim(4, ntime=2, nfreq=3)
    A = generate_proj(v.g_bar, v.V_model, remove=False)
    o = split_re_im(np.ravel(v.get_reduced_observed()))
    x = np.ravel(split_re_im((v.x)))
    p = np.dot(A, np.ravel(split_re_im((v.x))))

    print("Tol", np.max(abs(o-p)/o))
    
    # Generate FFTd data
    fft = fourier_operator(4)
 
    x = group_by_x(v.x)
    fft = np.fft.fft2(x)
    y = np.fft.ifft2(fft)
    
    # New AF operator.
    # The AF operator does this: receive a flat vector of floats
    # which can be reshaped to a grid of dimensions (nant, ntime, nfreq, 2)
    # The last 2 dimensions are the real/imag component of complex numbers
    # so create a complex-valued array of shape (nant, ntime, nfreq)
    # - Call np.fft.ifft2 on this array, which will do a 2-d inverse transform on the last two dimensions.
    # Result is x.
    # Move axis in x to produce an array of shape (ntime, nfreq, nant) which is complex valued.
    # Flatten the array (ravel) and split the complex values into real/imag types. Result: Vector x_flat.
    # Multiply this vector by the existing A operator, which will produce d, i.e. d = A x_flat
    
    # What does the A.T operator do. For a start, it must do things in the reverse order. That means 
    # multiplying the existing A.T operator by something that is like d. This will produce a vector
    # that is x-like, of shape flattened((ntime, nfreq, nant, 2)). Move axes and turn into a complex-valued
    # array of shape (nant, ntime, nfreq). Now we apply the transpose of the inverse Fourier operator,
    # which is the forward Fourier transform np.fft.fft2. The result of this is (nant, ntime, nfreq)
    # which can be flattened to (nant, ntime, nfreq, 2).
    
            
    
   
    
    # FFT the x values over frequency then test the recovery
    fft = fourier_operator(3)
    x = split_x_group_freq(v.x)
    for i in range(2):
        for j in range(8):
            x[i, j] = np.real

    
    FFT = multi_fft_operator(2, 3, 4)
    print(FFT.shape)
    
