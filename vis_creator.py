import numpy as np
import copy
import hickle as hkl
from pyuvdata import UVData, UVCal
from calcs import calc_visibility, split_re_im

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

    def __init__(self, nant, vis_model_sigma=1000, gain_sigma=3, x_sigma=0.1, obs_variance=1e4, level="approx", random_seed=None):
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
            
        level: str
            Specifies whether to use the gain offsets exactly or approximately.
            Options: [ "approx", "exact" ]
            "approx" means calculate baseline gains using gi*conj(gj)*(1+xi+conj(xj). 
            "exact" means calculate baseline gains using gi*conj(gj)*(1+xi)*(1+conj(xj)).
         
        """

        # check inputs 
        if nant <= 3 or vis_model_sigma <= 0 or gain_sigma <= 0 or x_sigma <= 0 \
                or level not in [ "approx", "exact" ]:
            raise ValueError("VisSim: Invalid input value")

        nvis = nant * (nant - 1) // 2
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Setup model and gains and x to generate observed visibilities

        re = np.random.normal(scale=vis_model_sigma, size=nvis)
        im = np.random.normal(scale=vis_model_sigma, size=nvis)
        V_model = re+1j*im

        re = np.random.normal(scale=gain_sigma, size=nant)
        im = np.random.normal(scale=gain_sigma, size=nant)
        g_bar = re+1j*im

        re = np.random.normal(scale=x_sigma, size=nant)
        im = np.random.normal(scale=x_sigma, size=nant)
        x = re+1j*im

        self.nant = nant
        self.nvis = nvis
        self.V_model = self.V_model_orig = V_model
        self.g_bar = self.g_bar_orig = g_bar
        self.x = self.x_orig = x
        self.level = level
        self.obs_variance = np.full(nvis, obs_variance)      # Baseline variances

        self.V_obs = self.get_simulated_visibilities()

        
        # Restore randomness
        if random_seed is not None: 
            np.random.seed()

    def get_simulated_visibilities(self):
        """
        Calculate V_model*gains*x for all baselines. 
        
        Returns
        -------
        
        array_like
            The new visibilities, indexed by baseline.
        """
        
        V = np.empty(self.nvis, dtype=type(self.V_model))
        k = 0
        for i in range(self.nant):
            for j in range(i+1, self.nant):
                V[k] = calc_visibility(self.level, self.g_bar[i], self.g_bar[j], 
                                       self.V_model[k], self.x[i], self.x[j])

                k += 1
        return V

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

        V_sim = self.get_simulated_visibilities()    # Used to do this a shorthand way
        
        return V_sim/self.V_model

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
        
        V = np.empty(self.V_obs.size, dtype=type(self.V_obs))
        k = 0
        for i in range(self.nant):
            for j in range(i+1, self.nant):     
                V[k] = self.V_obs[k]-self.V_model[k]*self.g_bar[i]*np.conj(self.g_bar[j])
                k += 1
        return V

    def get_calibrated_visibilities(self):
        """
        Divide the observed visibilities by the baseline gains.
        """
        
        return self.V_obs/self.get_baseline_gains()

    def get_quality(self):
        """
        Calculate the RMS of the real/imag values from get_diff_obs_sim()
        """
        
        rms = lambda x : np.sqrt(np.mean(x**2))

        dy = self.V_obs-self.get_simulated_visibilities()
        return rms(np.real(dy)), rms(np.imag(dy))

    def get_chi2(self):
        """
        The chi2 of the fit between V_obs and visibilities generated by V_model*gains*x
        """
        return np.sum(np.abs(self.V_obs-self.get_simulated_visibilities())**2/self.obs_variance)
    

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
    def __init__(self, file_root, time=0, freq=0, level="approx"):
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
        ants_to_baseline = []
        for i in range(nant):
            for j in range(i+1, nant):
                ants_to_baseline.append((i, j))
        assert uvdata.Nbls-nant == len(ants_to_baseline), \
                "Data does not contain expected number of baselines"
        nvis = len(ants_to_baseline)

        # Load V_obs
        V = np.zeros(nvis, dtype=np.complex64)
        for i, bl in enumerate(ants_to_baseline):
            V[i] = uvdata.get_data(bl[0], bl[1], "XX")[time][freq]

        # Get things out of calibration: model, redundant groups, weights

        fname = file_root+"_g_cal_dict.npz"
        print("Get model from", fname)
        cal = hkl.load(fname)

        baseline_groups = cal["all_reds"]
        redundant_groups = [ [] for bg in baseline_groups ]
        assert len(cal["v_omnical"].keys()) == len(baseline_groups), \
                "Number of baseline groups: "+str(len(baseline_groups))+ \
                " Number of model groups: "+str(len(cal["v_omnical"].keys()))
        V_model = np.zeros(nvis, dtype=np.complex64)
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
                    i = ants_to_baseline.index((bl[0], bl[1]))
                    V_model[i] = cal["v_omnical"][key][time][freq]
                except:
                    i = ants_to_baseline.index((bl[1], bl[0]))
                    V_model[i] = np.conj(cal["v_omnical"][key][time][freq])

                redundant_groups[group_index].append(i)

        assert np.min(np.abs(V_model)) > 0, "Model has missing values"

        # Get the weights used by calibration
        num_weights = len(list(cal["omni_meta"]["data_wgts"]))
        assert num_weights == V.size, "Not enough weights for baselines "+str(num_weights)+" "+str(V.size)
        weights = np.zeros(V.size)
        for key in cal["omni_meta"]["data_wgts"].keys():
            try:
                i = ants_to_baseline.index((key[0], key[1]))
            except:
                i = ants_to_baseline.index((key[1], key[0]))
            weights[i] = cal["omni_meta"]["data_wgts"][key][time][freq]


        # Get the gains generated from calibration

        fname = file_root+"_g_new.calfits"
        print("Get gains from", fname)
        uvc = UVCal()
        uvc.read_calfits(fname)
        g_bar = np.zeros(nant, dtype=np.complex64)
        for i in range(nant):
            g_bar[i] = uvc.get_gains(i)[0, 0]

        self.level = level
        self.nant = nant
        self.nvis = nvis
        self.V_obs = V
        self.V_model = V_model
        self.g_bar = g_bar
        self.x = np.zeros(g_bar.size, dtype=np.complex64)
        self.obs_variance = weights
        self.redundant_groups = redundant_groups

class VisTrue(VisSim):
    """
    Data from a non-redundant-pipeline simulation, not including calibration.
    """
    def __init__(self, file_root, time=0, freq=0, level="approx"):
        # Get model

        fname = file_root+".uvh5"
        print("Get true model from", fname)
        uvdata = UVData()
        uvdata.read_uvh5(fname)
        assert uvdata.Nants_data == uvdata.Nants_telescope, \
                "Not all antennas have data"
        nant = uvdata.Nants_data
        # Map antenna numbers to the vis index
        ants_to_baseline = []
        for i in range(nant):
            for j in range(i+1, nant):
                ants_to_baseline.append((i, j))
        assert uvdata.Nbls-nant == len(ants_to_baseline), \
                "Data does not contain expected number of baselines"
        nvis = len(ants_to_baseline)

        # Load model
        V_model = np.zeros(nvis, dtype=np.complex64)
        for i, bl in enumerate(ants_to_baseline):
            V_model[i] = uvdata.get_data(bl[0], bl[1], "XX")[time][freq]


        # Get true gains

        fname = file_root+".calfits"
        print("Get true gains from", fname)
        uvc = UVCal()
        uvc.read_calfits(fname)
        g_bar = np.zeros(nant, dtype=np.complex64)
        for i in range(nant):
            g_bar[i] = uvc.get_gains(i)[0, 0]

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
        V = np.zeros(nvis, dtype=np.complex64)
        for i, bl in enumerate(ants_to_baseline):
            V[i] = uvdata.get_data(bl[0], bl[1], "XX")[time][freq]

        self.level = level
        self.nant = nant
        self.nvis = nvis
        self.g_bar = g_bar
        self.V_model = V_model
        self.x = np.zeros(g_bar.size, dtype=np.complex64)
        self.V_obs = V
        self.obs_variance = np.full(V.size, 1)
        self.redundant_groups, _, _ = uvdata.get_redundancies()



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

def perturb_model(vis, vis_perturb_percent):
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
            choices = np.random.random(size=len(new_vis.V))-0.5
            V = np.where(choices <= 0, new_vis.V_model*(1-vis_perturb_percent/100.0), new_vis.V_model*(1+vis_perturb_percent/100.0))
            new_vis.V_model = V

    return new_vis


if __name__ == "__main__":
    v = VisSim(4, level="approx")
    v.print()
    v.get_baseline_gains()
    v.get_antenna_gains()
    v.get_reduced_observed()
    v.get_simulated_visibilities()
    v.get_calibrated_visibilities()
    v.get_quality()
    v.get_chi2()
    


