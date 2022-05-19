import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy
import hickle as hkl
from calcs import calc_visibility, split_re_im, unsplit_re_im, BlockMatrix, flatten_complex_2d

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
                 add_noise=False, level="approx", redundant=False):
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
                # The imag value for the last antenna should be 0
                im[-1] = 0
                x[time, freq] = re+1j*im
        
 
        bl_to_ants = []
        for i in range(nant):
            for j in range(i+1, nant):
                bl_to_ants.append((i, j))
                
        self.baseline_lengths = None

        self.nant = nant
        self.nvis = nvis
        self.ntime = ntime
        self.nfreq = nfreq
        self.bl_to_ants = bl_to_ants
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
        self.noise_added = False

        if add_noise:
            re = np.random.normal(scale=np.sqrt(obs_variance.real), size=self.nvis)
            im = np.random.normal(scale=np.sqrt(obs_variance.imag), size=self.nvis)
            self.V_obs += re+im*1j
            self.noise_added = True
        else: self.noise_added = False


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
        
        
    def plot_power_spectrum(self):
        if self.baseline_lengths is None:
            print("No baseline lengths in object, can't generate power spectrum")
            
        ps = np.zeros((len(self.redundant_groups), self.V_obs.shape[1]))
        nums = np.zeros(len(self.redundant_groups))
        vis = self.get_calibrated_visibilities()
          
        for time in range(self.V_obs.shape[0]):
            for group in range(len(self.redundant_groups)):
                for bl in self.redundant_groups[group]:
                    ps[group] += np.abs(np.fft.fftshift(np.fft.fft(vis[time, :, bl])))**2
                    nums[group] += 1
                    
                    
        for group in range(len(self.redundant_groups)):
            ps[group] /= nums[group]
            
        plt.matshow(ps.T[::-1], norm=LogNorm())
        plt.xlabel("Redundant groups")
        plt.ylabel("Delay")
        plt.colorbar()
                   
        
    def v_samp_V_unredundant(self):
        bm = BlockMatrix()
        bm.add(self.vis_redcal.model_projection, replicate=self.vis_redcal.ntime*self.vis_redcal.nfreq)
        redundant_projector = bm.assemble()   # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime
        
        V = np.dot(redundant_projector, np.ravel(split_re_im(self.vis_sampled.V_model)))
        V = unsplit_re_im(V)
        V = np.reshape(V, (self.vis_redcal.ntime, self.vis_redcal.nfreq, -1))
        return V

        

    
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

    def get_rms(self):
        diff = split_re_im(self.project_model())-split_re_im(self.get_calibrated_visibilities())
        return np.sqrt(np.mean((diff**2)))

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
        
        try: 
            UVData.__version__
        except: 
            from pyuvdata import UVData
        try: 
            UVCal.__version__
        except: 
            from pyuvdata import UVCal
            
            
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
        
        self.baseline_lengths = np.zeros(nvis)
        k = 0
        for i in range(nant):
            for j in range(i+1, nant):
                diff = uvdata.antenna_positions[i]-uvdata.antenna_positions[j]
                self.baseline_lengths[k] = np.sqrt(np.sum(diff**2))
                k += 1
  
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
        fname = file_root+"_nn.npz"
        noise_data = np.load(fname)["data_array"]
        uvdata.data_array = noise_data                    # Warning, reusing uvdata
        noise = np.zeros((ntime, nfreq, nvis), dtype=type(V[0, 0, 0]))
        for i, bl in enumerate(bl_to_ants):
            noise[:, :, i] = uvdata.get_data(bl[0], bl[1], "XX")[time_range[0]:time_range[1], freq_range[0]:freq_range[1]]
        if np.min(noise.imag) < 0: noise = np.conj(noise)        # In case uvdata has screwed around with the values

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
        self.noise_added = True
        

class VisTrue(VisSim):
    """
    Data from a non-redundant-pipeline simulation, not including calibration.
    """
    def __init__(self, file_root, time_range=None, freq_range=None, level="approx"):
        try: 
            UVData.__version__
        except: 
            from pyuvdata import UVData
        try: 
            UVCal.__version__
        except: 
            from pyuvdata import UVCal

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
        
        self.baseline_lengths = np.zeros(nvis)
        k = 0
        for i in range(nant):
            for j in range(i+1, nant):
                diff = uvdata.antenna_positions[i]-uvdata.antenna_positions[j]
                self.baseline_lengths[k] = np.sqrt(np.sum(diff**2))
                k += 1
        
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
        self.noise_added = True

 
        

class VisSampling:
    
    def __init__(self, vis, check_0_im=True, ignore_last_x_im=True):
        
        self.level = vis.level
        self.nant = vis.nant
        self.nvis = vis.nvis
        self.ntime = vis.ntime
        self.nfreq = vis.nfreq
        self.nvis = vis.nvis
        self.g_bar = vis.g_bar
        self.V_model = vis.V_model
        self.nmodel = vis.V_model.shape[2]
        self.x = vis.x
        self.V_obs = vis.V_obs
        self.obs_variance = vis.obs_variance
        self.redundant_groups = vis.redundant_groups
        self.bl_to_ants = vis.bl_to_ants
        self.model_projection = vis.model_projection
        self.noise_added = vis.noise_added
        self.ignore_last_x_im = ignore_last_x_im
                
        self.x_flat = np.ravel([ flatten_complex_2d(ant) for ant in np.moveaxis(self.x, 2, 0) ])
        assert np.max(np.abs(self.x_flat[-self.ntime*self.nfreq:])) == 0.0, np.max(np.abs(self.x_flat[:-self.ntime*self.nfreq]))
        
        if ignore_last_x_im: self.x_flat = self.x_flat[:-self.ntime*self.nfreq]
        
        self.V_model_flat = np.ravel(split_re_im(self.V_model))    # Now ordered by (time, freq, split re/im)

        self.reduced_observed_flat = np.ravel(split_re_im(vis.get_reduced_observed()))
        
        self.observed_flat = np.ravel(split_re_im(self.V_obs))
        
        self.group_mapping = self.map_to_redundant_group()

        
    def get_V_model(self, ti, fi, bl):
        """ vi is the index of complex vi """
        red_group = self.group_mapping[bl]
        return self.V_model_flat[ti*(self.nfreq*self.nmodel*2)+fi*self.nmodel*2+red_group*2] + \
                    self.V_model_flat[ti*(self.nfreq*self.nmodel*2)+fi*self.nmodel*2+red_group*2+1]*1j

    def get_x(self, ti, fi, xi):
        """ xi is the index of complex xi """
        return self.x_flat[xi*2*self.ntime*self.nfreq+ti*self.nfreq+fi] + \
                    self.x_flat[(xi*2+1)*self.ntime*self.nfreq+ti*self.nfreq+fi] * 1j
    
    def map_to_redundant_group(self):
        
        mapping = []
        red_index = 0
        for red in self.redundant_groups:
            for bl in red:
                mapping.append((bl, red_index))
            red_index += 1
        mapping.sort()

        return [ mapped[1] for mapped in mapping ]
        
    def generate_proj(self, g_bar, model, remove=True):

        def separate_real_imag(g0, g1, Vm):
            V = g0*g1*Vm
            Vij_re = (g0*g1*Vm).real
            Vij_im = (g0*g1*Vm).imag
            #print(Vij_re, -Vij_im, Vij_im, Vij_re); exit()

            return Vij_re, Vij_im

        # Generate the projection operator for each time/freq and merge
        bm = BlockMatrix()
        for time in range(g_bar.shape[0]):
            for freq in range(g_bar.shape[1]):
                proj = np.zeros((model.shape[2]*2, g_bar.shape[2]*2))
                k = 0
                for i in range(g_bar.shape[2]):
                    for j in range(i+1, g_bar.shape[2]):
                        re, im = separate_real_imag(g_bar[time, freq, i], np.conj(g_bar[time, freq, j]), model[time, freq, k])

                        proj[k*2,i*2] = re; proj[k*2,i*2+1] = -im
                        proj[k*2,j*2] = re; proj[k*2,j*2+1] = im

                        proj[k*2+1,i*2] = im; proj[k*2+1,i*2+1] = re
                        proj[k*2+1,j*2] = im; proj[k*2+1,j*2+1] = -re

                        k += 1


                if remove: bm.add(remove_x_im(proj))
                else: bm.add(proj)

        return bm.assemble()

    def generate_proj_sparse(self, g_bar, model, remove=True):

        def separate_real_imag(g0, g1, Vm):
            V = g0*g1*Vm
            Vij_re = (g0*g1*Vm).real
            Vij_im = (g0*g1*Vm).imag
            #print(Vij_re, -Vij_im, Vij_im, Vij_re); exit()

            return Vij_re, Vij_im

        # Generate the projection operator for each time/freq and merge
        mats = []
        for time in range(g_bar.shape[0]):
            for freq in range(g_bar.shape[1]):
                proj = np.zeros((model.shape[2]*2, g_bar.shape[2]*2))
                k = 0
                for i in range(g_bar.shape[2]):
                    for j in range(i+1, g_bar.shape[2]):
                        re, im = separate_real_imag(g_bar[time, freq, i], np.conj(g_bar[time, freq, j]), model[time, freq, k])

                        proj[k*2,i*2] = re; proj[k*2,i*2+1] = -im
                        proj[k*2,j*2] = re; proj[k*2,j*2+1] = im

                        proj[k*2+1,i*2] = im; proj[k*2+1,i*2+1] = re
                        proj[k*2+1,j*2] = im; proj[k*2+1,j*2+1] = -re

                        k += 1


                if remove: mats.append(scipy.sparse.coo_matrix(remove_x_im(proj)))
                else: mats.append(scipy.sparse.coo_matrix(proj))


        return scipy.sparse.block_diag(mats, format='coo')

    def generate_proj_sparse1(self, g_bar, model, remove=True):
        """
        remove: remove the x value that's unnecessary for sampling
        """

        def separate_real_imag(g0, g1, Vm):
            V = g0*g1*Vm
            Vij_re = (g0*g1*Vm).real
            Vij_im = (g0*g1*Vm).imag
            #print(Vij_re, -Vij_im, Vij_im, Vij_re); exit()

            return Vij_re, Vij_im

        def make_index_ant(ti, fi, xi):
            """ Flat index , taking into account the re/im split
            ai is the index of the x value after splitting,
            it is not the index of an antenna. Also take into account
            that there may be a missing x.
            """
            if remove: return ti*(nfreq*(nant*2-1))+fi*(nant*2-1)+xi
            return ti*(nfreq*nant*2)+fi*nant*2+xi

        def make_index_vis(ti, fi, vi):
            """ Flat index , taking into account the re/im split
            vi is the index of the visibility value after splitting,
            it is not the index of a baseline. 
            """
            return ti*(nfreq*nvis*2)+fi*nvis*2+vi

        def insert(ti, fi, r, c, val, ind):
            if not (remove and c == nant*2-1):    # Don't add the missing x
                row_inds[ind] = make_index_vis(ti, fi, r)
                col_inds[ind] = make_index_ant(ti, fi, c)
                data[ind] = val

                return ind+1

            else: return ind

        ntime = g_bar.shape[0]
        nfreq = g_bar.shape[1]
        nant = g_bar.shape[2]
        nvis = model.shape[2]

        nrows = ntime*nfreq*nvis*2
        if remove: ncols = ntime*nfreq*(nant*2-1)
        else: ncols = ntime*nfreq*nant*2

        # Generate the projection operator 

        nvalues = ntime*nfreq*nvis*8
        if remove: nvalues -= ntime*nfreq*(nant-1)*2 # A measure of how many times the last antenna gets involved in a baseline, and then re/im (x2)

        # For the list of data points
        row_inds = np.zeros(nvalues)
        col_inds = np.zeros(nvalues)
        data = np.zeros(nvalues)
        value_index = 0

        for time in range(ntime):
            for freq in range(nfreq):

                # Go through the baselines
                bl = 0
                for ant1 in range(nant):
                    for ant2 in range(ant1+1, nant):
                        re, im = separate_real_imag(g_bar[time, freq, ant1], np.conj(g_bar[time, freq, ant2]), model[time, freq, bl])

                        value_index = insert(time, freq, bl*2, ant1*2, re, value_index)
                        value_index = insert(time, freq, bl*2, ant1*2+1, -im, value_index)
                        value_index = insert(time, freq, bl*2, ant2*2, re, value_index)                   
                        value_index = insert(time, freq, bl*2, ant2*2+1, im, value_index)
                        value_index = insert(time, freq, bl*2+1, ant1*2, im, value_index)
                        value_index = insert(time, freq, bl*2+1, ant1*2+1, re, value_index)                
                        value_index = insert(time, freq, bl*2+1, ant2*2, im, value_index)
                        value_index = insert(time, freq, bl*2+1, ant2*2+1, -re, value_index)

                        bl += 1


        return scipy.sparse.coo_matrix((data, (row_inds, col_inds)), shape=(nrows, ncols))

    def generate_proj_sparse2(self):
        """
        The columns of the operator correspond to x values laid out in
        order of x re/im first, for each one of these there is a block of values
        of size ntime*nfreq flattened
        """

        def separate_real_imag(g0, g1, Vm):
            V = g0*g1*Vm
            Vij_re = (g0*g1*Vm).real
            Vij_im = (g0*g1*Vm).imag
            #print(Vij_re, -Vij_im, Vij_im, Vij_re); exit()

            return Vij_re, Vij_im

        def make_index_ant(ti, fi, xi):
            """ Flat index , taking into account the re/im split
            xi is the index of the x value after splitting,
            it is not the index of an antenna. Also take into account
            that there may be a missing x, hence factors (nant*2-1).

            """

            return xi*ntime*nfreq+ti*nfreq+fi

        def make_index_vis(ti, fi, vi):
            """ Flat index , taking into account the re/im split
            vi is the index of the visibility value after splitting,
            it is not the index of a baseline. 
            """

            return ti*(nfreq*nvis*2)+fi*nvis*2+vi

        def insert(ti, fi, r, c, val, ind):
            row_inds[ind] = make_index_vis(ti, fi, r)
            col_inds[ind] = make_index_ant(ti, fi, c)
            data[ind] = val


        ntime = self.g_bar.shape[0]
        nfreq = self.g_bar.shape[1]
        nant = self.g_bar.shape[2]
        nvis = self.V_obs.shape[2]
        
        nrows = ntime*nfreq*nvis*2
        ncols = ntime*nfreq*nant*2

        # Generate the projection operator 

        nvalues = ntime*nfreq*nvis*8    # Num values that will go into the matrix

        # For the list of data points
        row_inds = np.zeros(nvalues, dtype=np.int)
        col_inds = np.zeros(nvalues, dtype=np.int)
        data = np.zeros(nvalues, dtype=np.int)
        value_index = 0

        for time in range(ntime):
            for freq in range(nfreq):

                # Go through the baselines
                bl = 0
                for ant1 in range(nant):
                    for ant2 in range(ant1+1, nant):
                        re, im = separate_real_imag(self.g_bar[time, freq, ant1], np.conj(self.g_bar[time, freq, ant2]), self.get_V_model(time, freq, bl))

                        insert(time, freq, bl*2, ant1*2, re, value_index); value_index += 1
                        insert(time, freq, bl*2, ant1*2+1, -im, value_index); value_index += 1
                        insert(time, freq, bl*2, ant2*2, re, value_index); value_index += 1                   
                        insert(time, freq, bl*2, ant2*2+1, im, value_index); value_index += 1
                        insert(time, freq, bl*2+1, ant1*2, im, value_index); value_index += 1
                        insert(time, freq, bl*2+1, ant1*2+1, re, value_index); value_index += 1                
                        insert(time, freq, bl*2+1, ant2*2, im, value_index); value_index += 1
                        insert(time, freq, bl*2+1, ant2*2+1, -re, value_index); value_index += 1

                        bl += 1

        if self.ignore_last_x_im:
            # Knock off the last block of time/freq for the last imaginary x value
            # These are the last ntime*nfreq columns, leaving columns up to (nant*2-1)*ntime*nfreq
            row_inds = row_inds[col_inds<(nant*2-1)*ntime*nfreq]
            data = data[col_inds<(nant*2-1)*ntime*nfreq]
            col_inds = col_inds[col_inds<(nant*2-1)*ntime*nfreq]
            ncols = (nant*2-1)*ntime*nfreq
            
        return scipy.sparse.coo_matrix((data, (row_inds, col_inds)), shape=(nrows, ncols))

    def generate_proj_sparse3(self, remove_last_x_im=False):
        """
        The columns of the operator correspond to x values laid out in
        order of x re/im first, for each one of these there is a block of values
        of size ntime*nfreq flattened
        """

        def separate_real_imag(g0, g1, Vm):
            V = g0*g1*Vm
            Vij_re = (g0*g1*Vm).real
            Vij_im = (g0*g1*Vm).imag
            #print(Vij_re, -Vij_im, Vij_im, Vij_re); exit()

            return Vij_re, Vij_im

        def make_index_ant(ti, fi, xi):
            """ Flat index , taking into account the re/im split
            xi is the index of the x value after splitting,
            it is not the index of an antenna. Also take into account
            that there may be a missing x, hence factors (nant*2-1).

            """

            return xi*ntime*nfreq+ti*nfreq+fi

        def make_index_vis(ti, fi, vi):
            """ Flat index , taking into account the re/im split
            vi is the index of the visibility value after splitting,
            it is not the index of a baseline. 
            """
            return ti*(nfreq*nvis*2)+fi*nvis*2+vi

        def insert(ti, fi, r, c, val, ind):
            row_inds[ind] = make_index_vis(ti, fi, r)
            col_inds[ind] = make_index_ant(ti, fi, c)
            data[ind] = val



        ntime = self.g_bar.shape[0]
        nfreq = self.g_bar.shape[1]
        nant = self.g_bar.shape[2]
        nvis = self.V_model.shape[2]

        nrows = ntime*nfreq*nvis*2
        ncols = ntime*nfreq*nant*2

        # Generate the projection operator 

        nvalues = ntime*nfreq*nvis*8    # Num values that will go into the matrix

        # For the list of data points
        row_inds = np.zeros(nvalues, dtype=np.int)
        col_inds = np.zeros(nvalues, dtype=np.int)
        data = np.zeros(nvalues, dtype=np.int)
        value_index = 0

        for time in range(ntime):
            for freq in range(nfreq):

                # Go through the baselines
                bl = 0
                for ant1 in range(nant):
                    for ant2 in range(ant1+1, nant):
                        re, im = separate_real_imag(self.g_bar[time, freq, ant1], np.conj(self.g_bar[time, freq, ant2]), self.get_V_model(time, freq, bl))

                        insert(time, freq, bl*2, ant1*2, re, value_index); value_index += 1
                        insert(time, freq, bl*2, ant1*2+1, -im, value_index); value_index += 1
                        insert(time, freq, bl*2, ant2*2, re, value_index); value_index += 1                   
                        insert(time, freq, bl*2, ant2*2+1, im, value_index); value_index += 1
                        insert(time, freq, bl*2+1, ant1*2, im, value_index); value_index += 1
                        insert(time, freq, bl*2+1, ant1*2+1, re, value_index); value_index += 1                
                        insert(time, freq, bl*2+1, ant2*2, im, value_index); value_index += 1
                        insert(time, freq, bl*2+1, ant2*2+1, -re, value_index); value_index += 1

                        bl += 1

        if remove_last_x_im:
            # Knock off the last block of time/freq for the last imaginary x value
            # These are the last ntime*nfreq columns, leaving columns up to (nant*2-1)*ntime*nfreq
            row_inds = row_inds[col_inds<(nant*2-1)*ntime*nfreq]
            data = data[col_inds<(nant*2-1)*ntime*nfreq]
            col_inds = col_inds[col_inds<(nant*2-1)*ntime*nfreq]
            ncols = (nant*2-1)*ntime*nfreq
            
        return scipy.sparse.coo_matrix((data, (row_inds, col_inds)), shape=(nrows, ncols))

    
    def generate_m_proj_sparse1(self):
        """
        Generate the projection operator from gain parameter vectors 
        to visibilities as a CSR sparse array.

        This is a block diagonal array. Each block is for a given time and 
        frequency. Within each block, the projection operator is constructed 
        using the appropriate values of g_bar etc. for each antenna pair.
        """
        
        def separate_terms(gi, gj, xi, xj):
            v = gi*np.conj(gj)*(1+xi+np.conj(xj))
            return v.real, v.imag
        

        def make_index_vis(ti, fi, vi):
            """ Flat index , taking into account the re/im split
            vi is the index of the visibility value after splitting,
            it is not the index of a baseline. 
            """
            return ti*(self.nfreq*self.nvis*2)+fi*self.nvis*2+vi

        def insert(ti, fi, r, c, val, ind):
            rows[ind] = make_index_vis(ti, fi, r)
            cols[ind] = make_index_vis(ti, fi, c)
            data[ind] = val

                
        nrows = self.ntime*self.nfreq*self.nvis*2
        ncols = nrows
        
        nvalues = self.ntime*self.nfreq*self.nvis*4
        value_index = 0

        # Generate the projection operator 
        rows = np.zeros(nvalues)
        cols = np.zeros(nvalues)
        data = np.zeros(nvalues)

        # Loop over times and frequencies
        for time in range(self.ntime):
            for freq in range(self.nfreq):

                bl = 0
                for ant1 in range(self.nant):
                    for ant2 in range(ant1+1, self.nant):
                   
                        term1, term2 = separate_terms(self.g_bar[time, freq, ant1], 
                                                      self.g_bar[time, freq, ant2], 
                                                      self.x[time, freq, ant1], 
                                                      self.x[time, freq, ant2])

                        # Put them in the right place 
                        insert(time, freq, bl*2, bl*2, term1, value_index); value_index+=1
                        insert(time, freq, bl*2, bl*2+1, -term2, value_index); value_index+=1
                        insert(time, freq, bl*2+1, bl*2, term2, value_index); value_index+=1
                        insert(time, freq, bl*2+1, bl*2+1, term1, value_index); value_index+=1
                        
                        bl += 1



        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(nrows, ncols))





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



if __name__ == "__main__":
    
    np.random.seed(99)
    # Check d = A x
    v = VisSim(6, ntime=2, nfreq=3)

    v = VisSampling(v)
    print(v.x[1, 0, 2]); 
    print(v.get_x(1, 0, 2)); exit()

    A = v.generate_proj_sparse2(v.g_bar, v.V_model)


    o = v.reduced_observed_flat
    x = v.x_flat

    p = A.dot(x)
    for i in range(o.size):
        print(o[i], p[i])

    print("Tol", np.max(abs(o-p)/o))
    
