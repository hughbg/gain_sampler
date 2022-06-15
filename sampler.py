#from matplotlib import use; use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from vis_creator import VisSim, VisCal, VisTrue, VisSampling
from calcs import print_statistics, split_re_im, unsplit_re_im, BlockMatrix, remove_x_im, restore_x_im, is_diagonal
from fourier_ops import FourierOps
from resources import Resources
import corner
import copy
import numpy as np
import scipy.linalg, scipy.sparse, scipy.sparse.linalg
import sys


        
def load_from_files(dirname):
    import hickle

    sampler = hickle.load(dirname+"/sampler.hkl")
    samples = np.load(dirname+"/samples.npz")
    for param in [ "x", "g", "V" ]:
        sampler.samples[param] = samples[param]         
        
    # Create an object containing the best fit
    #best_vals = sampler.bests(method=sampler.best_type)
    #sampler.vis_sampled.x = best_vals["x"]
    #sampler.vis_sampled.V_model = best_vals["V"]
    
    return sampler



        
class Sampler:
    
    """
    A class that loads simulations, runs sampling of x and V, and has functions for plotting results.

    Parameters
    ----------
    niter : int
        How many iterations of Gibbs sampling to execute.
    burn_in : float
        The percentage of samples to throw away at the beginning.
        Must be 0-99.
    seed : float
        Set this as a random seed, so that sampling runs can be repeated.
    random_the_long_way : bool
        If true, draw random sampled for Gibbs sampling using contrained realization equations.
        Otherwise, use the equivalent multivariate Gaussian and draw samples from that.
    best_type: str
        After sampling, the "best" samples are selected and used as the "result" of sampling.
        This parameter indicates how to select the best samples. There are three options:
        "mean": Get the mean value of each x, V sampled value.
        "peak": Form histograms and get the peak of the histogram, for each x, V sampled value.
        "ml": Find the set of samples that has the highest likelihood. This is different from
            the previous two options because it operates on the complete set of parameters at each
            sampling step, intead of treating each parameter individually.
    """
    
    def __init__(self, niter=1000, burn_in=10, seed=None, random_the_long_way=False, use_conj_grad=True, 
                best_type="mean", best_use="x", best_measure="rms", report_every=1000):


        if seed is not None:
            np.random.seed(seed)
            
        assert niter > 0, "niter is invalid"
        assert burn_in >=0 and burn_in < 100, "burn_in must be 0-99"
        n_samples_not_burned = niter-(niter*burn_in)//100
        assert n_samples_not_burned > 0, "burn_in will leave no samples"
        self.niter = niter
        self.burn_in = burn_in
        self.random_the_long_way = random_the_long_way
        self.use_conj_grad = use_conj_grad
        if best_type in [ "mean", "median", "peak", "ml" ]:
            self.best_type = best_type
        else:
            raise ValueError("best_type has invalid value: "+str(best_type))
        if best_use in [ "x", "g" ]:
            self.best_use = best_use
        else:
            raise ValueError("best_use has invalid value: "+str(best_use))
        if best_measure in [ "rms", "likelihood" ]:
            self.best_measure = best_measure
        else:
            raise ValueError("best_measure has invalid value: "+str(best_measure))

        self.report_every = report_every
        self.seed = seed
        
        self.gain_degeneracies_fixed = False
        
        self.samples = None
    
    def load_nr_sim(self, file_root, time_range=None, freq_range=None, remove_redundancy=False, 
                    with_redcal=True, initial_solve_for_x=False):
        """
        Load a simulation from the Non-Redundant pipeline.

        Parameters
        ----------
        file_root : str
            The pattern that will be used to find all the files associated with the simulation.
            Example: /scratch3/users/hgarsden/catall/calibration_points/viscatBC
            The file_root string will be extended to form actual file names.
        time_range : tuple of 2 integers or None
            (start_time_index, end_time_index)
            Indexes into the simulation that specify a chunk of time to use from the
            simulation. The range is Python style so it is up to but not including
            end_time_index.
            If None, then all times are used.
        freq_range: tuple of 2 integers or None
            Like time_range but specifies a chunk of frequencies.
        remove_redundancy: bool
            If True: remove the redundant sets of baselines and place each baseline 
            into its own redundant set.
        initial_solve_for_x : bool
            If True: Use a generalized least squares solver to generate x offsets that
            are applied to gains in the redcal calibrated solution. Hence improving the
            redcal solution.

        """
        

        print("Loading NR sim from", file_root)
        self.vis_true = VisTrue(file_root, time_range=time_range, freq_range=freq_range)
        if with_redcal:
            self.vis_redcal = VisCal(file_root, time_range=time_range, freq_range=freq_range, 
                                     remove_redundancy=remove_redundancy) 
        else:
            self.vis_redcal = self.vis_true

        if initial_solve_for_x:
            self.vis_redcal.x = self.vis_redcal.initial_vals.x = gls_solve(self.vis_redcal)
            
            
        # Indexes into the original simulated data
        self.time_range = time_range
        self.freq_range = freq_range
        
        self.file_root = file_root
        if self.random_the_long_way:
            self.fops = FourierOps(self.vis_redcal.nfreq, self.vis_redcal.ntime, self.vis_redcal.nant)

            
    def load_sim(self, nant, ntime=1, nfreq=1, initial_solve_for_x=False, **kwargs):

        self.vis_redcal = VisSim(nant, ntime=ntime, nfreq=nfreq, **kwargs)
        self.vis_true = self.vis_redcal
        if self.random_the_long_way:
            self.fops = FourierOps(ntime, nfreq, nant)

        self.file_root = ""
        if self.random_the_long_way:
            self.fops = FourierOps(self.vis_redcal.nfreq, self.vis_redcal.ntime, self.vis_redcal.nant)
        
            
    def set_S_and_V_prior(self, S, V_mean, Cv):

        N_diag = np.zeros(0)
        for time in range(self.vis_redcal.ntime):
            for freq in range(self.vis_redcal.nfreq):
                N_diag = np.append(N_diag, split_re_im(self.vis_redcal.obs_variance[time][freq]))

        Cv_diag = np.zeros(0)
        for time in range(self.vis_redcal.ntime):
            for freq in range(self.vis_redcal.nfreq):
                Cv_diag = np.append(Cv_diag, Cv)


        self.S = S
        self.V_mean = np.ravel(split_re_im(V_mean))
        self.Cv_diag = Cv_diag
        self.N_diag = N_diag
        
    def nant(self):
        return self.vis_redcal.nant
    
    def nvis(self):
        return self.vis_redcal.nvis
    
                   
    def run(self):
        if not hasattr(self,"vis_true"):
            raise RuntimeError("No sim loaded. Can't sample.")
            
        print("Running sampling")
        sampled_x = np.zeros((self.niter, self.vis_redcal.ntime*self.vis_redcal.nfreq*(self.vis_redcal.nant*2-1)),
                            dtype=type(self.vis_redcal.x[0, 0, 0]))   
                                                                # -1 because there'll be a missing imaginary value
        sampled_V = np.zeros((self.niter, self.vis_redcal.ntime*self.vis_redcal.nfreq*len(self.vis_redcal.redundant_groups)*2),
                             dtype=type(self.vis_redcal.V_model[0, 0, 0].real))
 
        v_x_sampling = VisSampling(self.vis_redcal, ignore_last_x_im=True)  
        v_x_sampling.s_flat = self.fops.F_v(v_x_sampling.x_flat)
        try:
            v_x_sampling.s_flat = v_x_sampling.s_flat[self.S.usable_modes]
        except:
            raise RuntimeError("S prior is of wrong size")


        v_model_sampling = VisSampling(self.vis_redcal) 

        new_x = v_model_sampling.x         # Initialize
        
        resources = Resources()
        
        # Take num samples
        for i in range(self.niter):
            
            # Use the sampled x to change the model sampling distribution, and take a sample
            v_model_sampling.x = new_x
            if self.random_the_long_way:
                sample = self.V_random_draw_sparse(v_model_sampling)
            else:
                v_dist_mean, v_dist_covariance = self.new_model_distribution(v_model_sampling)
                sample = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)
            
            v_model_sampling.V_model_flat = sample
            sampled_V[i] = sample

            # Use the sampled model to change the x sampling distribution, and take a sample
            v_x_sampling.V_model_flat = sample
            if self.random_the_long_way:
                s_sample, x_sample = self.x_random_draw_sparse(v_x_sampling)
            else:
                x_dist_mean, x_dist_covariance = self.new_x_distribution(v_x_sampling)
                sample = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  
            
            v_x_sampling.x_flat = x_sample
            v_x_sampling.s_flat = s_sample

            sampled_x[i] = x_sample
            
            if i%self.report_every == 0: 
                print("Iter", i)
                resources.report()

            sys.stdout.flush()
            
            
        sampled_x = sampled_x[(self.niter*self.burn_in)//100:]
        # Turn each x into a 3-D array (ntime, nfreq, nant)
        extra_x_im = np.zeros((sampled_x.shape[0], self.vis_redcal.ntime*self.vis_redcal.nfreq))
        sampled_x = np.append(sampled_x, extra_x_im, axis=1)  
        sampled_x = np.reshape(sampled_x, (sampled_x.shape[0], self.vis_redcal.nant*2, self.vis_redcal.ntime, self.vis_redcal.nfreq))         
        # Now combine the grids into complex numbers
        sampled_x = sampled_x[:, 0::2]+sampled_x[:, 1::2]*1j
        sampled_x = np.moveaxis(sampled_x, 1, 3)
        
        
        sampled_V = sampled_V[(self.niter*self.burn_in)//100:]
        # Turn each V into a 3-D array (ntime, nfreq, nvis) 
        sampled_V = unsplit_re_im(sampled_V)
        sampled_V = sampled_V.reshape((sampled_V.shape[0], self.vis_redcal.ntime, self.vis_redcal.nfreq, self.vis_redcal.V_model.shape[2]))

        
        sampled_gains = np.zeros_like(sampled_x)
        for i in range(sampled_gains.shape[0]):
            sampled_gains[i] = self.vis_redcal.g_bar*(1+sampled_x[i])
        
        self.samples = {
            "x" : sampled_x,
            "g" : sampled_gains,
            "V" : sampled_V
        }
        
        # Create an object containing the best fit
        # Create an object containing the best fit
        self.vis_sampled = copy.deepcopy(self.vis_redcal)    
        best_vals = self.bests(self.best_use, method=self.best_type, measure=self.best_measure)
        if self.best_use == "x":
            self.vis_sampled.x = best_vals["x"]
        else:
            self.vis_sampled.g_bar = best_vals["g"]
            self.vis_sampled.x.fill(0)
        self.vis_sampled.V_model = best_vals["V"]
    
    def select_samples_by_time_freq(self, what, time=None, freq=None):
        assert what in [ "x", "g", "V" ], "Invalid sample specification"

        if time is None and freq is not None:
            #assert False, "Both a time and frequency must be specified"
            samples = self.samples[what][:, :, freq:freq+1, :]
        elif time is not None and freq is None:
            #assert False, "Both a time and frequency must be specified"
            samples = self.samples[what][:, time:time+1, :, :]
        elif time is not None and freq is not None:
            samples = self.samples[what][:, time:time+1, freq:freq+1, :]
        else:
            samples = self.samples[what]

        return samples
            
    def select_values_by_time_freq(self, values, time=None, freq=None):

        if time is None and freq is not None:
            #assert False, "Both a time and frequency must be specified"
            v = values[:, freq:freq+1, :]
        elif time is not None and freq is None:
            #assert False, "Both a time and frequency must be specified"
            v = values[time:time+1, :, :]
        elif time is not None and freq is not None:
            v = values[time:time+1, freq:freq+1, :]
        else:
            v = values
            
        return v
            
    def remove_unsampled_x(self, x):
        samples_x = split_re_im(x) 
        if len(x.shape) == 3:
            samples_x = np.delete(samples_x, samples_x.shape[2]-1, axis=2) 
        else: 
            samples_x = np.delete(samples_x, samples_x.shape[3]-1, axis=3)     # Remove unsampled x
        samples_x = samples_x.reshape(samples_x.shape[0], -1)
            
        return samples_x
            
    def reform_x_from_samples(self, sampled_x, shape):
        x = sampled_x.reshape(shape[0], shape[1], shape[2]*2-1)
        x = restore_x_im(x)
        return unsplit_re_im(x)
    
    def plot_marginals(self, parameter, cols, time=None, freq=None, which=[ "True", "Redcal", "Sampled" ],
                      limit=None):
        def plot_hist(a, fname, label, sigma_prior, other_vals, index):
            hist, bin_edges = np.histogram(a, bins=max(len(a)//50, 5))
            bins = (bin_edges[1:]+bin_edges[:-1])/2

            sigma = np.std(a-np.mean(a))

            plt.subplot(rows, cols, index)
            plt.plot(bins, hist, "k", linewidth=0.6)
            for key in other_vals:
                plt.axvline(other_vals[key][0], color=other_vals[key][1], label=key, linewidth=0.6)
            #plt.title(label+" sigma: "+str(round(sigma,2))+" sigma_prior: "+str(round(sigma_prior,2)))
            plt.title(label)
            plt.legend()
            
        assert isinstance(parameter, str), "marginal parameter must be string"
        if len(which) > 0:
            for w in which:
                assert w in [ "True", "Redcal", "Sampled" ], "Invalid results to plot: "+str(w)

        print("Plot marginals")
        
        
        if parameter == "x":
            samples_x = self.select_samples_by_time_freq("x", time, freq)
            samples_x = self.remove_unsampled_x(samples_x)
            if limit is not None:
                samples_x = samples_x[:, :limit]

            num_plots = samples_x.shape[1]
            if num_plots%cols == 0: rows = num_plots//cols
            else: rows = num_plots//cols+1

            true_x = np.ravel(split_re_im(self.select_values_by_time_freq(self.vis_true.g_bar, time, freq)/
                                 self.select_values_by_time_freq(self.vis_redcal.g_bar, time, freq)-1))
            redcal_x = np.ravel(split_re_im(self.select_values_by_time_freq(self.vis_redcal.x, time, freq)))
            best_sampled_x = np.ravel(split_re_im(self.select_values_by_time_freq(self.vis_sampled.x, time, freq)))

            for i in range(samples_x.shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {}
                if "True" in which: other_vals["True"] = ( true_x[i], "green" )
                if "Redcal" in which: other_vals["Redcal"] = ( redcal_x[i], "red" )
                if "Sampled" in which: other_vals["Sampled"] = ( best_sampled_x[i], "blue" )

                plot_hist(samples_x[:, i], part+"_x_"+str(i//2), part+"(x_"+str(i//2)+")", None, other_vals, i+1)
        elif parameter == "g":
            samples_g = self.select_samples_by_time_freq("g", time, freq).reshape(self.samples["g"].shape[0], -1)
            samples_g = split_re_im(samples_g)
            if limit is not None:
                samples_g = samples_g[:, :limit]

            
            num_plots = samples_g.shape[1]
            if num_plots%cols == 0: rows = num_plots//cols
            else: rows = num_plots//cols+1
            
            true_g = split_re_im(np.ravel(self.select_values_by_time_freq(self.vis_true.g_bar, time, freq)))
            redcal_g = split_re_im(np.ravel(self.select_values_by_time_freq(self.vis_redcal.g_bar, time, freq)))
            best_sampled_g = split_re_im(np.ravel(self.select_values_by_time_freq(self.vis_sampled.get_antenna_gains(), time, freq)))
            
            for i in range(samples_g.shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {}
                if "True" in which: other_vals["True"] = ( true_g[i], "green" )
                if "Redcal" in which: other_vals["Redcal"] = ( redcal_g[i], "red" )
                if "Sampled" in which: other_vals["Sampled"] = ( best_sampled_g[i], "blue" )

                plot_hist(samples_g[:, i], part+"_g_"+str(i//2), part+"(g_"+str(i//2)+")", None, other_vals, i+1)
                
        elif parameter == "V":
            samples_V = self.select_samples_by_time_freq("V", time, freq).reshape(self.samples["V"].shape[0], -1)
            samples_V = split_re_im(samples_V)
            if limit is not None:
                samples_V = samples_V[:, :limit]

            
            num_plots = samples_V.shape[1]
            if num_plots%cols == 0: rows = num_plots//cols
            else: rows = num_plots//cols+1
            
            true_V = split_re_im(np.ravel(self.select_values_by_time_freq(self.vis_true.V_model, time, freq)))
            redcal_V = split_re_im(np.ravel(self.select_values_by_time_freq(self.vis_redcal.V_model, time, freq)))
            best_sampled_V = split_re_im(np.ravel(self.select_values_by_time_freq(self.vis_sampled.V_model, time, freq)))
            for i in range(samples_V.shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {}
                if "True" in which: other_vals["True"] = ( true_V[i], "green" )
                if "Redcal" in which: other_vals["Redcal"] = ( redcal_V[i], "red" )
                if "Sampled" in which: other_vals["Sampled"] = ( best_sampled_V[i], "blue" )

                plot_hist(samples_V[:, i], part+"_V_g"+str(i//2), part+"(V_g"+str(i//2)+")", None, other_vals, i+1)
        else:
            raise ValueError("Invalid spec for plot_marginals")
            
        plt.tight_layout()
        plt.savefig("x.png")
        
    def plot_one_over_time_freq(self, param, sample_index, param_index, time=None, freq=None, plot_to=None):
        assert param in [ "x", "g", "V" ], "param must be x or g or V"
        assert ((time is None and freq is not None) or (time is not None and freq is None)) \
            and not (time is None and freq is None), "Invalid specification of time/freq"
        
        samples = self.samples[param][sample_index]     # (ntime, nfreq, nparam)
        if time is not None: 
            samples = samples[time, :, param_index]
            title = "Param: "+param+". Time "+str(time)+" over freq. Sample: "+str(sample_index)
            xlabel = "Freq index"
        else: 
            samples = samples[:, freq, param_index]
            title = "Param: "+param+". Freq "+str(freq)+" over time. Sample: "+str(sample_index)
            xlabel = "Time index"
        
        plt.plot(samples.real, label="Real")
        plt.plot(samples.imag, label="Imag")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Value")
        plt.legend()
        
        if plot_to is not None:
            plt.savefig(plot_to+".pdf")

        
    def plot_corner(self, parameters, time, freq, threshold=0.0, baselines=None, plot_to=None):
        assert parameters == ["x", "V"] or parameters == ["g", "V"], "corner plot needs x,V or g,V"
        assert threshold >= 0, "threshold must be >= 0"
        assert not (threshold > 0.0 and baselines is not None), "Can't specify threshold and baselines at the same time"
               
        
        print("Plot corner")
        
        data_packet = self.assemble_data(parameters, time, freq)       # Puts both together flat

        part = lambda i : "re" if i%2==0 else "im"
       
        name = "x" if "x" in parameters else "g"
        labels = [ r"$"+part(i)+"("+name+"_"+str(i//2)+")$" for i in range(data_packet["x_or_g_len"]) ]+  \
                    [ r"$"+part(i)+"(V_{g"+str(i//2)+")}$" for i in range(data_packet["V_len"]) ]
        
        if threshold > 0.0:
            m = np.corrcoef(data_packet["data"], rowvar=False)
            m[np.triu_indices(m.shape[0])] = 0

            num_x_left = 0

            m_indices = np.argwhere(np.abs(m) > threshold)

            data_indices = np.unique(np.ravel(m_indices))
            
            to_delete = np.arange(m.shape[0])[np.isin(np.arange(m.shape[0]), data_indices, invert=True)]
            labels = [ labels[i] for i in range(len(labels)) if i not in to_delete ]
            num_x_left = data_packet["x_or_g_len"]-np.where(to_delete<data_packet["x_or_g_len"])[0].size
         
            data_packet["data"] = np.delete(data_packet["data"], to_delete, axis=1)
            data_packet["x_or_g_len"] = num_x_left
            data_packet["V_len"] = data_packet["data"].shape[1]-num_x_left
            
            assert data_packet["data"].shape[0] > 0, "No parameters to plot"
            
        if baselines is not None:
            # baselines are in redundant groups. Find the which group baselines are in
            groups = []

            for i in range(len(self.vis_sampled.redundant_groups)):
                for bl in baselines:
                    if bl in self.vis_sampled.redundant_groups[i] and i not in groups:
                       groups.append(i)

            assert len(groups) > 0, "Can't find baselines "+str(baselines)     
            
            # Find the antennas
            ants = []
            for bl in baselines:
                bl_ants = self.vis_sampled.bl_to_ants[bl]
                if bl_ants[0] not in ants: ants.append(bl_ants[0])
                if bl_ants[1] not in ants: ants.append(bl_ants[1])
           
            # Now work out which x or g indexes (called xgs) and V indexes (called Vs), based on
            # them being split into re/im
            xgs = [ a*2 for a in ants ] + [ a*2+1 for a in ants ]
            xgs = [ x for x in xgs if x < data_packet["x_or_g_len"] ]  # Careful about mising x imaginary value

            Vs = [ v*2+data_packet["x_or_g_len"] for v in groups ] + [ v*2+1+data_packet["x_or_g_len"] for v in groups ] 

            to_delete = np.arange(data_packet["data"].shape[1])[np.isin(np.arange(data_packet["data"].shape[1]), xgs+Vs, invert=True)]

            labels = [ labels[i] for i in range(len(labels)) if i not in to_delete ]
            num_x_left = data_packet["x_or_g_len"]-np.where(to_delete<data_packet["x_or_g_len"])[0].size

            data_packet["data"] = np.delete(data_packet["data"], to_delete, axis=1)
            data_packet["x_or_g_len"] = num_x_left
            data_packet["V_len"] = data_packet["data"].shape[1]-num_x_left
            
            assert num_x_left <= data_packet["V_len"]*4, "Too many values "+str(num_x_left)+" > "+str(data_packet["V_len"]*4)    # N baselines can only map up to N*2 ants which are split re/im
            assert data_packet["data"].shape[1] == len(labels), "Incorrect number of labels "+str(data_packet["data"].shape[1])+ " vs "+str(len(labels))
            assert data_packet["data"].shape[0] > 0, "No parameters to plot"
            

            
        print(name+" values:", str(data_packet["x_or_g_len"])+",", "V values:", data_packet["V_len"])

        figure = corner.corner(data_packet["data"], labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)
        
        axes = np.array(figure.axes).reshape((data_packet["data"].shape[1], data_packet["data"].shape[1]))
        for i in range(data_packet["data"].shape[1]):
            for j in range(data_packet["data"].shape[1]):
                ax = axes[i, j]
                if i < data_packet["x_or_g_len"]:
                    color = "blue"
                else:
                    if j < data_packet["x_or_g_len"]: color = "red"
                    else:
                        color = "darkgreen"
                for key in ax.spines: ax.spines[key].set_color(color)
        
        """
        # Loop over the diagonal
        true_gains = split_re_im(self.vis_true.g_bar)
        true_model = split_re_im(self.vis_true.V_model)
        orig = np.concatenate((true_gains, true_model))

        for i in range(orig.size):
            #print(i, orig[i])
            ax = axes[i, i]
            ax.axvline(orig[i], color="gold", linewidth=0.5)

        true_gains = split_re_im(self.vis_redcal.g_bar)
        true_model = split_re_im(self.vis_redcal.V_model)
        orig = np.concatenate((true_gains, true_model))

        for i in range(orig.size):
            #print(i, orig[i])
            ax = axes[i, i]
            ax.axvline(orig[i], color="red", linewidth=0.5)
        """
        plt.tight_layout()
        
        if plot_to is not None:
            plt.savefig(plot_to+".pdf")
   
        #return figure
        
    def plot_trace(self, parameter, time=None, freq=None, index=None, plot_to=None):
        assert isinstance(parameter, str), "trace parameter must be string"
        assert parameter in [ "x", "g", "V" ], "Unknown parameter: "+parameter
        
        print("Plot trace")
        
        if parameter == "V": parameter = "Vg"
        
        data = self.select_samples_by_time_freq(parameter, time, freq)
        if parameter == "x":
            data = self.remove_unsampled_x(data)
        else:
            data = split_re_im(data)
        data = data.reshape((data.shape[0], -1))

        sample_range = np.arange((self.burn_in*self.niter)//100, self.niter, 1)
        if index is None:
            for i in range(data.shape[1]):
                plt.plot(sample_range, data[:, i])
        else:
            plt.plot(sample_range, data[:, index])
            
        plt.xlabel("Sample iteration")
        plt.ylabel(parameter)
        plt.title("Traces for "+parameter)
        
        if plot_to is not None:
            plt.savefig(plot_to+".pdf")

                 
    def print_covcorr(self, parameters, time=None, freq=None, stat="corr", threshold=0.0, list_them=False, count_them=False):
        assert len(parameters) == 2, "covariance needs x,V or g,V"
        assert stat == "cov" or stat == "corr", "Specify cov or corr for matrix"
        assert threshold >= 0
        
        print("Print covcorr")
        
        part = lambda i : "re" if i%2==0 else "im"

        data_packet = self.assemble_data(parameters, time, freq)
        m = np.cov(data_packet["data"], rowvar=False) if stat == "cov" else np.corrcoef(data_packet["data"], rowvar=False) 
        if count_them:
            np.fill_diagonal(m, 0)
            m[np.triu_indices(m.shape[0])] = 0
            print(np.sum(np.where(abs(m) > threshold, 1, 0)))
            return

        name = "x" if "x" in parameters else "g"
        labels = [ part(i)+"("+name+"_"+str(i//2)+")" for i in range(data_packet["x_or_g_len"]) ]+  \
                    [ part(i)+"(V_g"+str(i//2)+")" for i in range(data_packet["V_len"]) ]
        
        if not list_them: 
            print(" "*len(labels[0]), end="\t")
            for l in labels: print(l, end="\t")
            print()
        for i in range(m.shape[0]):                  # m should be square
            if not list_them: print(labels[i], end="\t")
            for j in range(m.shape[1]):
                if abs(m[i, j]) > threshold:
                    if list_them: 
                        if i > j: print(labels[i]+","+labels[j], "\t", np.round(m[i, j], 2))
                    else: print(np.round(m[i, j], 2), end="\t")
                else: 
                    if not list_them: 
                        print("-----", end="\t")
            if not list_them: print()
                
                
    def examine_all_for_gaussianity(self, skew_limit=0.1, kurtosis_limit=0.2):
        """
        Find the skew_limit, kurtosis_limit by running some tests on lots of sample draws
        of the same sample size used in Sampler.
        """
        
        for param in [ "x", "g", "V" ]:
            vals = split_re_im(self.samples[param])
            for t in range(vals.shape[1]):
                for f in range(vals.shape[2]):
                    for i in range(vals.shape[3]):
                        #print(t, f, i, np.mean(x[:, t, f, i]), np.std(x[:, t, f, i]), scipy.stats.skew(x[:, t, f, i]),
                              #scipy.stats.kurtosis(x[:, t, f, i]))
                        mean = np.mean(vals[:, t, f, i])
                        sigma = np.std(vals[:, t, f, i])
                        #if param == "x" and np.abs(mean) > 3*sigma:
                        #    print("x", t, f, i, "shifted beyond sigma")

                        skew = scipy.stats.skew(vals[:, t, f, i])
                        kurtosis = scipy.stats.skew(vals[:, t, f, i])
                        if np.abs(skew) > skew_limit:
                            print(param, t, f, i, "high skew", skew)
                        if np.abs(kurtosis) > kurtosis_limit:
                            print(param, t, f, i, "high kurtosis", kurtosis)

            
            
    def plot_covcorr(self, parameters, time=None, freq=None, stat="corr", threshold=0.0, 
                     hist=False, log=False, plot_to=None):
        assert len(parameters) == 2, "covariance needs x,V or g,V"
        assert stat == "cov" or stat == "corr", "Specify cov or corr for matrix"
        assert threshold >= 0
        
        part = lambda i : "re" if i%2==0 else "im"
        
        if stat == "corr":
            print("Plotting correlation matrix")
        else:
            print("Plotting covariance matrix")
            
        param_tag = "x" if "x" in parameters else "g"


        data_packet = self.assemble_data(parameters, time, freq)
        print(param_tag+" values:", str(data_packet["x_or_g_len"])+",", "V values:", data_packet["V_len"])
        
        m = np.cov(data_packet["data"], rowvar=False) if stat == "cov" else np.corrcoef(data_packet["data"], rowvar=False) 
                
        if hist:
            
            histogram, bin_edges = np.histogram(m, bins=100)
            plt.plot((bin_edges[:-1]+bin_edges[1:])/2, histogram)
            if log: plt.yscale("log")
            plt.xlabel(stat.capitalize())
            plt.ylabel("Count")
            if plot_to is not None: plt.savefig(plot_to+".pdf")
            return
        
        
        print("Matrix size", m.shape)

        np.fill_diagonal(m, 0)
        m[np.triu_indices(m.shape[0])] = 0
        if threshold > 0.0: m[np.logical_and(m>-threshold,m<threshold)] = 0
            
        print_statistics(m)

        if log: m = np.abs(m)
        plt.figure()
        ax = plt.gca()
        if log:
            im = ax.matshow(m, cmap="RdBu", norm=LogNorm())
        else:
            im = ax.matshow(m, cmap="RdBu")
        nx = data_packet["x_or_g_len"]
        nv =  data_packet["V_len"]
        x=plt.xticks([nx//2, nx+nv//2], [param_tag, "V"], fontsize=15)
        x=plt.yticks([nx//2, nx+nv//2], [param_tag, "V"], fontsize=15)
        plt.plot([-0.5, nx+nv-0.5], [nx-0.5, nx-0.5], "k", linewidth=0.6)
        plt.plot([nx-0.5, nx-0.5], [-0.5, nx+nv-0.5],  "k", linewidth=0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        if stat == "cov": 
            plt.title("Covariance matrix "+str(parameters))
        else:
            plt.title("Correlation matrix "+str(parameters))
            
        if plot_to is not None:
            plt.savefig(plot_to+".pdf")
            
    def plot_sample_means(self, parameters, time=None, freq=None):
        assert len(parameters) <= 3, "Too many parameters requested"
        for p in parameters:
            assert p in [ "x", "g", "V" ], "Invalid parameter: "+p
            
        print("Plot sample means")
            
        for i, p in enumerate(parameters):
            data = self.select_samples_by_time_freq(p, time, freq).reshape(self.samples[p].shape[0], -1)
            data = split_re_im(data)
            plt.subplot(len(parameters), 1, i+1)
            plt.plot(range(data.shape[1]), np.mean(data, axis=0), label="Means", linewidth=0.6)
            x=plt.xticks(range(0, data.shape[1], 4), range(0, data.shape[1], 4))
            plt.xlabel("Order")
            plt.legend()
            plt.title("Sample means \""+p+"\" (split real/imag) in order")

        
        plt.tight_layout()
        
    def print_sample_stats(self, parameter, time=None, freq=None):
        assert parameter in [ "x", "g", "V" ], "print_sample_stats needs x or g or V"
        
        print("Stats for", parameter, "samples\n")
        
        re_im = lambda index: "re" if index%2 == 0 else "im"
        
        samples = split_re_im(self.select_samples_by_time_freq(parameter, time, freq))
        
        print("  Time  Freq   Parameter      Mean       Variance       Sigma")
        for t in range(samples.shape[1]):
            for f in range(samples.shape[2]):
                for i in range(samples.shape[3]):
                    print("{:5d}".format(t), "{:5d}".format(f),  "  ", "{:8}".format(re_im(i)+"("+parameter+"_"+str(i//2)+")"),  
                          "{:12f}".format(np.mean(samples[:, t, f, i])), 
                           "{:12f}".format(np.var(samples[:, t, f, i])),  "{:12f}".format(np.std(samples[:, t, f, i])))
  
    def plot_results(self, time=None, freq=None): 
        
        def print_fit_stats(name, x, y):
            new_series, info = np.polynomial.polynomial.polyfit(x, y, 1, full=True)
            print(name, "Slope:", new_series[-1], "Error:", info[0][0])

   
        self.fix_degeneracies()
        print("Plot results")
        
        v_true = self.select_values_by_time_freq(self.vis_true.V_model, time, freq)
        v_redcal = self.select_values_by_time_freq(self.vis_redcal.get_calibrated_visibilities(), time, freq)
        v_sampled = self.select_values_by_time_freq(self.vis_sampled.get_calibrated_visibilities(), time, freq)

        v_true = split_re_im(np.ravel(v_true))
        v_redcal = split_re_im(np.ravel(v_redcal))
        v_sampled = split_re_im(np.ravel(v_sampled))
        
        plt.subplot(2, 1, 1)
        order = np.abs(v_true).argsort()
        plt.plot(np.abs(v_true[order]), 
                 np.abs(v_true[order]), "k", linewidth=0.6,  label="1:1")
        plt.plot(np.abs(v_true[order]), 
                 np.abs(v_redcal[order]), "r", linewidth=0.6,  label="Redcal")
        print_fit_stats("Redcal Amp", np.abs(v_true[order]), np.abs(v_redcal[order]))
        plt.plot(np.abs(v_true[order]), 
                 np.abs(v_sampled[order]), "b", linewidth=0.6,  label="Sampled")
        print_fit_stats("Sampled Amp", np.abs(v_true[order]), np.abs(v_sampled[order]))
        plt.legend()
        plt.xlabel("V_true amplitude")
        plt.ylabel("Amplitude")
        plt.title("Calibrated visibilities (Amplitude)")
        
        plt.subplot(2, 1, 2)
        order = np.angle(v_true).argsort()
        plt.plot(np.angle(v_true.astype(np.complex64)[order]), 
                 np.angle(v_true.astype(np.complex64)[order]), "k", linewidth=0.6,  label="1:1")
        plt.plot(np.angle(v_true.astype(np.complex64)[order]), 
                 np.angle(v_redcal.astype(np.complex64)[order]), "r", linewidth=0.6,  label="Redcal")
        print_fit_stats("Redcal Phase", np.angle(v_true[order]), np.angle(v_redcal[order]))
        plt.plot(np.angle(v_true.astype(np.complex64)[order]), 
                 np.angle(v_sampled.astype(np.complex64)[order]), "b", linewidth=0.6,  label="Sampled")
        print_fit_stats("Sampled Phase", np.angle(v_true[order]), np.angle(v_sampled[order]))
        plt.legend()
        plt.xlabel("V_true phase")
        plt.ylabel("Phase")
        plt.title("Calibrated visibilities (Phase)")
        plt.tight_layout()
        
    def plot_gains(self, time=None, freq=None, sigma=3, plot_to=None):
        def normalize_phases(phases):
            # Make sure phases are between -pi, pi
            phases = np.where(phases>3*np.pi/2, phases-2*np.pi, phases)
            phases = np.where(phases<-3*np.pi/2, phases+2*np.pi, phases)
            return phases
        
        def get_g_error_bars():
            BOTTOM = 0
            TOP = 1
            
            sampled_gains = self.select_samples_by_time_freq("g", time, freq).reshape(self.samples["g"].shape[0], -1)                                    
            g_limits_amp = np.zeros((sampled_gains.shape[1], 2))    # 2 beacause bottom and top
            g_limits_phase = np.zeros((sampled_gains.shape[1], 2))
            
            # Get the error bars. For each g, get the range based on sigma.
            for i in range(sampled_gains.shape[1]):       # Loop over antennas
                amps = np.abs(sampled_gains[:, i])
                m = np.mean(amps)
                s = np.std(amps)
                g_limits_amp[i, BOTTOM] = m-sigma*s
                g_limits_amp[i, TOP] = m+sigma*s
                
                phases = normalize_phases(np.angle(sampled_gains[:, i]))
                m = np.mean(phases)
                s = np.std(phases)
                g_limits_phase[i, BOTTOM] = m-sigma*s
                g_limits_phase[i, TOP] = m+sigma*s
                
                
            return g_limits_amp, g_limits_phase
        
        
        #g_bar = self.v_sampled.g_bar
        #print(get_x_error_bars())
        #exit()
        
        print("Plot gains")
        
        error_amp, error_phase = get_g_error_bars()

        g_true = self.select_values_by_time_freq(self.vis_true.get_antenna_gains(), time, freq)
        g_redcal = self.select_values_by_time_freq(self.vis_redcal.get_antenna_gains(), time, freq)
        g_sampled = self.bests("g", time=time, freq=freq, method=self.best_type, measure=self.best_measure)["g"]

        g_true = np.ravel(g_true)
        g_redcal = np.ravel(g_redcal)
        g_sampled = np.ravel(g_sampled)

        plt.subplot(2, 1, 1)
                                      
        # Amplitude
                                
        # 1. true. The gains are 1? Can't remember why
        plt.plot(range(g_true.size), np.abs(g_true), color="green", linewidth=0.6, label="g_true")

        # 2. The redcal gains as they are given to us by redcal.
        plt.plot(range(g_true.size), np.abs(g_redcal), "r", linewidth=0.6, label="g_redcal")

        # 3. Sampled gains based on sampled x. 
        plt.plot(range(g_true.size), np.abs(g_sampled), "b", linewidth=0.6, label="g_sampled")
        
        # Error bars

        assert error_amp.shape[0] == g_true.size, str(error_amp.shape[0])+" "+str(g_true.size)
        for i in range(error_amp.shape[0]):
            plt.plot([i, i], [ error_amp[i][0], error_amp[i][1] ], "lightblue")
        plt.legend()
        plt.title("Gain amplitudes")
        plt.xlabel("Antenna")
        plt.ylabel("Amplitude")
        
        # Phase

        plt.subplot(2, 1, 2)

        # 1. V true. The gains are 1? Can't remember why
        plt.plot(range(g_true.size), np.unwrap(np.angle(g_true)), color="green", linewidth=0.6, label="g_true")

        # 2. The redcal gains as they are given to us by redcal.
        plt.plot(range(g_true.size), np.unwrap(np.angle(g_redcal.astype(np.complex64))), "r", linewidth=0.6, label="g_redcal")

        # 3. The sampled gains, actually x is sampled. 
        plt.plot(range(g_true.size), np.unwrap(np.angle(g_sampled.astype(np.complex64))), "b", linewidth=0.6, label="g_sampled")
        
        # Error bars
        assert error_phase.shape[0] == g_true.size
        for i in range(error_phase.shape[0]):
            plt.plot([i, i], [ error_phase[i][0], error_phase[i][1] ], "lightblue")
        plt.legend()
        plt.title("Gain phases")
        plt.xlabel("Antenna")
        plt.ylabel("Phase (rad)")
        plt.tight_layout()
                        
        if plot_to is not None:
            plt.savefig(plot_to+".pdf")
       
        

    def assemble_data(self, parameters, time, freq, remove_0_x=True):
        assert parameters == ["x", "V"] or parameters == ["g", "V"], "assemble_data needs x,V or g,V"
        
        data_packet = {}
        
        what = "x" if "x" in parameters else "g"            
        samples = self.select_samples_by_time_freq(what, time, freq)
        if remove_0_x and "x" in parameters:
            samples = self.remove_unsampled_x(samples)
        else:
            samples = split_re_im(samples) 
            
        data_packet["data"] = samples.reshape((samples.shape[0], -1))
        data_packet["x_or_g_len"] = data_packet["data"].shape[1]
             
        # Now tack V on
        samples = self.select_samples_by_time_freq("V", time, freq).reshape(self.samples["V"].shape[0], -1)
        samples = split_re_im(samples)        
        data_packet["data"] = np.concatenate((data_packet["data"], samples), axis=1)
        data_packet["V_len"] = samples.shape[1]

        return data_packet
               
    def standard_random_draw(self, size):

        return np.random.normal(size=size)
    
        
    def test_distributions(self):
        sampled_x = np.zeros((self.niter, self.vis_redcal.nant*2-1))       # -1 because there'll be a missing imaginary value
        sampled_V = np.zeros((self.niter, self.V_mean.size*2))

        for i in range(self.niter):
            if self.random_the_long_way:
                sampled_x[i] = self.x_random_draw(self.vis_redcal)
                sampled_V[i] = self.V_random_draw(self.vis_redcal)
            else:
                x_dist_mean, x_dist_covariance = self.new_x_distribution(self.vis_redcal)
                sampled_x[i] = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  

                v_dist_mean, v_dist_covariance = self.new_model_distribution(self.vis_redcal)
                sampled_V[i] = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)    
                
        self.sampled_x = sampled_x
        self.sampled_V = sampled_V

    def r_draw(self, d, A, S, N_diag, previous_x):
        
            
        """
        does the dynamic range fixup.
        print(v.nvis, v.nant, v.ntime, v.nfreq, S.shape, N.shape, d.shape, A.shape)
        
        #6 4 2 3 (42, 42) (72, 72) (72,) (72, 42)
        S: (nant*2)-1 * ntime*nfreq
        ant: ntime*nfreq*(2*nant-1)
        d: ntime*nfreq*nvis*2
        N: ntime*nfreq*nvis*2  square
        A: d x ant
        """
       
        # [1 + sqrt(S)(A.T N-1 A)sqrt(S)]x = sqrt(S) A.TN-1 A d + random + sqrt(S)A.T sqrt(N_inv) random

        usable_modes = S[1][0]
        S_values = S[0]
        sqrt_S = np.sqrt(S_values)

        N_diag_inv = 1/N_diag
        
        # Work on RHS
        sqrt_S_AFT_Ninv_d = (sqrt_S*self.fops.F_v(np.dot(A.T, N_diag_inv*d)))[usable_modes]
        sqrt_S_AFT_sqrt_Ninv_omega = (sqrt_S*self.fops.F_v(np.dot(A.T, np.sqrt(N_diag_inv)*self.standard_random_draw(d.size))))[usable_modes]
        rhs = sqrt_S_AFT_Ninv_d+self.standard_random_draw(usable_modes.size)+sqrt_S_AFT_sqrt_Ninv_omega

        # Work on the LHS
        
        def matvec(v):
            # Put the x values in the usable mode places. Select them out after we are done
            return v+(sqrt_S*self.fops.F_v(np.dot(A.T, N_diag_inv*np.dot(A, self.fops.F_inv_fft(sqrt_S*place_usable_modes(v))))))[usable_modes]
            
                 
        def operator():                            
            shape = (usable_modes.size, usable_modes.size)
        
            return scipy.sparse.linalg.LinearOperator(shape, matvec=matvec, dtype=np.float)
        
        op = operator()

        x_proxy, info = scipy.sparse.linalg.cg(op, rhs)             # Conjugate gradient for stability
        assert info == 0, str(info)
        
        s = sqrt_S*place_usable_modes(x_proxy)
        x = self.fops.F_inv_fft(s)       # Just converts s to x

        return x

    def r_draw_sparse(self, d, A, S, N_diag, previous_s):
        
        def place_usable_modes(vals):
            modes = np.zeros_like(S_values)
            modes[usable_modes] = vals
            return modes
            
        """
        does the dynamic range fixup.
        print(v.nvis, v.nant, v.ntime, v.nfreq, S.shape, N.shape, d.shape, A.shape)
        
        #6 4 2 3 (42, 42) (72, 72) (72,) (72, 42)
        S: (nant*2)-1 * ntime*nfreq
        ant: ntime*nfreq*(2*nant-1)
        d: ntime*nfreq*nvis*2
        N: ntime*nfreq*nvis*2  square
        A: d x ant
        """
       
        # [1 + sqrt(S)(A.T N-1 A)sqrt(S)]x = sqrt(S) A.TN-1 A d + random + sqrt(S)A.T sqrt(N_inv) random

        usable_modes = S[1][0]
        S_values = S[0]
        sqrt_S = np.sqrt(S_values)

        N_diag_inv = 1/N_diag
        
        # Work on RHS
        sqrt_S_AFT_Ninv_d = (sqrt_S*self.fops.F_v((A.T).dot(N_diag_inv*d)))[usable_modes]
        sqrt_S_AFT_sqrt_Ninv_omega = (sqrt_S*self.fops.F_v((A.T).dot(np.sqrt(N_diag_inv)*self.standard_random_draw(d.size))))[usable_modes]
        rhs = sqrt_S_AFT_Ninv_d+self.standard_random_draw(usable_modes.size)+sqrt_S_AFT_sqrt_Ninv_omega

        # Work on the LHS
        
        def matvec(v):
            # Put the x values in the usable mode places. Select them out after we are done
            return v+(sqrt_S*self.fops.F_v((A.T).dot(N_diag_inv*A.dot(self.fops.F_inv_fft(sqrt_S*place_usable_modes(v))))))[usable_modes] 
            
                 
        def operator():                            
            shape = (usable_modes.size, usable_modes.size)
        
            return scipy.sparse.linalg.LinearOperator(shape, matvec=matvec, dtype=np.float)
        
        op = operator()

        x_proxy, info = scipy.sparse.linalg.cg(op, rhs, x0=previous_s)             # Conjugate gradient for stability
        assert info == 0, str(info)
        
        s = sqrt_S*place_usable_modes(x_proxy)
        x = self.fops.F_inv_fft(s)       # Just converts s to x

        return x

                           
    def random_draw(self, first_term, A, S_diag, N_diag, previous_V):
        N_diag_inv = 1/N_diag
        S_diag_inv = 1/S_diag
        S_sqrt = np.sqrt(S_diag)
        A_N_A = np.dot(A.T*N_diag_inv, A).T       # If N is a vector


        rhs = np.multiply(S_sqrt, first_term)
        rhs += self.standard_random_draw(A.shape[1])
        rhs += np.multiply(S_sqrt, np.dot(A.T, np.multiply(np.sqrt(N_diag_inv), self.standard_random_draw(A.shape[0]))))

        bracket_term = np.eye(S_diag.size)
        bracket_term += (A_N_A.T*S_sqrt).T*S_sqrt

        if self.use_conj_grad:
            x, info = scipy.sparse.linalg.cg(bracket_term, rhs, x0=previous_V)
            assert info == 0
        else: x = np.dot(np.linalg.inv(bracket_term), rhs)

        x = np.multiply(S_sqrt, x)

        return x

    def random_draw_sparse(self, b, A, S_diag, N_diag, previous_V):

        # Vectors of matrix diagonals
        N_diag_inv = 1.0/N_diag
        S_diag_inv = 1.0/S_diag
        S_sqrt = np.sqrt(S_diag)

        # Square matrix operator
        AT_N_A = ((A.T).multiply(N_diag_inv) @ A)

        # Draw unit Gaussian random values
        omega0 = self.standard_random_draw(A.shape[1])
        omega1 = self.standard_random_draw(A.shape[0])

        # RHS vector, multiplied through by diagonal S^1/2
        # Construct RHS vector (operation ordering #1)
        #rhs = np.multiply(S_sqrt, b) # weight b by S^1/2
        #rhs += omega0 # omega (unit normal) random vector
        #rhs += np.multiply(S_sqrt, 
        #                   (A.T).dot( np.multiply(np.sqrt(N_diag_inv), 
        #                                          omega1) ))

        # Construct RHS vector (operation ordering #2)
        rhs = np.multiply(S_sqrt, 
                          b + (A.T).dot( np.multiply(np.sqrt(N_diag_inv), 
                                                     omega1) ))
        rhs += omega0

        # Construct linear operator
        bracket_term = scipy.sparse.identity(S_diag.size) \
                     + ((AT_N_A.multiply(S_sqrt)).T).multiply(S_sqrt)

        # Perform sparse CG solve
        x, info = scipy.sparse.linalg.cg(bracket_term, rhs, x0=previous_V)
        assert info == 0

        # Multiply solution by S^1/2 and return
        x = np.multiply(S_sqrt, x)

        return x

       
    def x_random_draw_sparse(self, v):

        # [1 + sqrt(S)(A.T N-1 A)sqrt(S)]x = sqrt(S) A.TN-1 A d + random + sqrt(S)A.T sqrt(N_inv) random

        def place_usable_modes(vals):
            modes = np.zeros_like(S_values)
            modes[usable_modes] = vals
            return modes

        d = v.reduced_observed_flat

        usable_modes = self.S.usable_modes
        S_values = self.S.S_values
        sqrt_S = np.sqrt(S_values)
        
        A = v.generate_proj_sparse2()  
        assert S_values.size == A.shape[1], "S prior is of wrong size"

        N_diag_inv = 1.0/self.N_diag
        
        # Work on RHS
        sqrt_S_AFT_Ninv_d = (sqrt_S*self.fops.F_v((A.T).dot(N_diag_inv*d)))[usable_modes]
        sqrt_S_AFT_sqrt_Ninv_omega = (sqrt_S*self.fops.F_v((A.T).dot(np.sqrt(N_diag_inv)*self.standard_random_draw(d.size))))[usable_modes]
        rhs = sqrt_S_AFT_Ninv_d+self.standard_random_draw(usable_modes.size)+sqrt_S_AFT_sqrt_Ninv_omega

        # Work on the LHS
        
        def matvec(v):
            # Put the x values in the usable mode places. Select them out after we are done
            return v+(sqrt_S*self.fops.F_v((A.T).dot(N_diag_inv*A.dot(self.fops.F_inv_fft(sqrt_S*place_usable_modes(v))))))[usable_modes] 
            
                 
        def operator():                            
            shape = (usable_modes.size, usable_modes.size)
        
            return scipy.sparse.linalg.LinearOperator(shape, matvec=matvec, dtype=np.float)
        
        op = operator()

        previous_s = v.s_flat

        x_proxy, info = scipy.sparse.linalg.cg(op, rhs, x0=previous_s)             # Conjugate gradient for stability
        assert info == 0, str(info)

        s = sqrt_S*place_usable_modes(x_proxy)
        x = self.fops.F_inv_fft(s)       # Just converts s to x

        return (s[usable_modes], x)
     
    def V_random_draw(self, v):
        A = self.generate_m_proj(v)   # Square matrix of shape nvis*2*ntime*nfreq
        bm = BlockMatrix()
        bm.add(v.model_projection, replicate=v.ntime*v.nfreq)
        redundant_projector = bm.assemble()   # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime
        A = np.dot(A, redundant_projector)    # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime


        V_mean = split_re_im(np.ravel(self.V_mean))
        d = split_re_im(np.ravel(v.V_obs))
        
        previous_V = split_re_im(np.ravel(v.V_model))

        """
        print("V ---------")
        print("d", np.mean(d), np.std(d))
        print("A", np.mean(A), np.std(A))
        print("Cv", np.mean(self.Cv_diag), np.std(self.Cv_diag))
        print("N_diag", np.mean(self.N_diag), np.std(self.N_diag))
        """
        
        # Vectors of matrix diagonals
        N_diag_inv = 1.0/self.N_diag
        Cv_sqrt = np.sqrt(Cv_diag)

        # Square matrix operator
        AT_N_A = ((A.T).multiply(N_diag_inv) @ A)

        # Draw unit Gaussian random values
        omega0 = self.standard_random_draw(A.shape[1])
        omega1 = self.standard_random_draw(A.shape[0])

        # RHS vector, multiplied through by diagonal S^1/2
        # Construct RHS vector (operation ordering #1)
        #rhs = np.multiply(S_sqrt, b) # weight b by S^1/2
        #rhs += omega0 # omega (unit normal) random vector
        #rhs += np.multiply(S_sqrt, 
        #                   (A.T).dot( np.multiply(np.sqrt(N_diag_inv), 
        #                                          omega1) ))

        # Construct RHS vector (operation ordering #2)
        rhs = np.multiply(Cv_sqrt, 
                          b + (A.T).dot( np.multiply(np.sqrt(N_diag_inv), 
                                                     omega1) ))
        rhs += omega0

        # Construct linear operator
        bracket_term = scipy.sparse.identity(Cv_diag.size) \
                     + ((AT_N_A.multiply(S_sqrt)).T).multiply(Cv_sqrt)

        # Perform sparse CG solve
        V, info = scipy.sparse.linalg.cg(bracket_term, rhs, x0=previous_V)
        assert info == 0

        # Multiply solution by S^1/2 and return
        V = np.multiply(S_sqrt, x)

        return V
    
    def V_random_draw_sparse(self, v):

        A = v.generate_m_proj_sparse1()   

        all_model_projections = [ scipy.sparse.coo_matrix(v.model_projection) for i in range(v.ntime*v.nfreq) ]
        redundant_projector = scipy.sparse.block_diag(all_model_projections, format='coo')
        A = A.dot(redundant_projector)    # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime
        
        d = v.observed_flat
        previous_V = v.V_model_flat

        # RHS of linear system (vector)
        b = (A.T).dot( np.multiply(1/self.N_diag, d)) \
          + np.multiply(1/self.Cv_diag, self.V_mean)

        # Solve linear system

        # Vectors of matrix diagonals
        N_diag_inv = 1.0/self.N_diag
        Cv_sqrt = np.sqrt(self.Cv_diag)

        # Square matrix operator
        AT_N_A = ((A.T).multiply(N_diag_inv) @ A)

        # Draw unit Gaussian random values
        omega0 = self.standard_random_draw(A.shape[1])
        omega1 = self.standard_random_draw(A.shape[0])

        # RHS vector, multiplied through by diagonal S^1/2
        # Construct RHS vector (operation ordering #1)
        #rhs = np.multiply(S_sqrt, b) # weight b by S^1/2
        #rhs += omega0 # omega (unit normal) random vector
        #rhs += np.multiply(S_sqrt, 
        #                   (A.T).dot( np.multiply(np.sqrt(N_diag_inv), 
        #                                          omega1) ))

        # Construct RHS vector (operation ordering #2)
        rhs = np.multiply(Cv_sqrt, 
                          b + (A.T).dot( np.multiply(np.sqrt(N_diag_inv), 
                                                     omega1) ))
        rhs += omega0

        # Construct linear operator
        bracket_term = scipy.sparse.identity(Cv_sqrt.size) \
                     + ((AT_N_A.multiply(Cv_sqrt)).T).multiply(Cv_sqrt)

        # Perform sparse CG solve
        V, info = scipy.sparse.linalg.cg(bracket_term, rhs, x0=previous_V)
        assert info == 0

        # Multiply solution by S^1/2 and return
        V = np.multiply(Cv_sqrt, V)

        return V
    
    def distribute_redundant_V_samples(self):
        bm = BlockMatrix()
        bm.add(v.model_projection, replicate=v.ntime*v.nfreq)
        redundant_projector = bm.assemble()   # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime
        print(samples["V"].shape)
        


    def new_x_distribution(self, v):
        # The model has been updated so get a new distribution.
        # If S is set to None and the model is never changed then
        # the mean of the x distribution will be the GLS solution.

        A = generate_proj(v.g_bar, v.project_model())  # depends on model
        
        bm = BlockMatrix()
        for time in range(v.ntime):
            for freq in range(v.nfreq):
                bm.add((split_re_im(v.obs_variance[time][freq])))
        N = bm.assemble()
        
        d = split_re_im(np.ravel(v.get_reduced_observed()))                     
        #d = split_re_im(v.get_reduced_observed1()) 
        
        bm = BlockMatrix()
        for time in range(v.ntime):
            for freq in range(v.nfreq):
                bm.add((self.S_diag))
        S = bm.assemble()
        

        N_inv = np.linalg.inv(N)
        S_inv = np.linalg.inv(S)

        term1 = np.dot(A.T, np.dot(N_inv, A))
        term2 = np.dot(A.T, np.dot(N_inv, d))
        dist_mean = np.dot(np.linalg.inv(S_inv+term1), term2)
        dist_covariance = np.linalg.inv(S_inv+term1)

        return dist_mean, dist_covariance

    
    def separate_terms(self, gi, gj, xi, xj):
        v = gi*np.conj(gj)*(1+xi+np.conj(xj))
        return v.real, v.imag

    def generate_m_proj(self, vis):
        bm = BlockMatrix()
        for time in range(vis.ntime):
            for freq in range(vis.nfreq):
                proj = np.zeros((vis.nvis*2, vis.nvis*2))
                k = 0
                for i in range(vis.nant):
                    for j in range(i+1, vis.nant):
                        term1, term2 = self.separate_terms(vis.g_bar[time, freq, i], vis.g_bar[time, freq, j], 
                                                            vis.x[time, freq, i], vis.x[time, freq, j])
                        # Put them in the right place in the bigger matrix
                        proj[k*2, k*2] = term1
                        proj[k*2, k*2+1] = -term2
                        proj[k*2+1, k*2] = term2
                        proj[k*2+1, k*2+1] = term1

                        k += 1
                        
                bm.add(proj)

        # Test - first line equal to second
        #print(split_re_im(vis.get_simulated_visibilities()))
        #print(np.dot(proj, split_re_im(vis.V_model))); exit()

        return bm.assemble()
    
   
    def generate_m_proj_sparse(self, vis):
        """
        Generate the projection operator from gain parameter vectors 
        to visibilities as a CSR sparse array.

        This is a block diagonal array. Each block is for a given time and 
        frequency. Within each block, the projection operator is constructed 
        using the appropriate values of g_bar etc. for each antenna pair.
        """
        proj = np.zeros((vis.nvis*2, vis.nvis*2), dtype=np.float64)
        mats = []

        # Loop over times and frequencies
        for time in range(vis.ntime):
            for freq in range(vis.nfreq):
                #proj_dok = scipy.sparse.dok_matrix((vis.nvis*2, vis.nvis*2), dtype=np.float64)
                # Loop over antenna pairs to construct projection matrix
                proj *= 0.0 # Set all elements to zero
                k = 0
                for i in range(vis.nant):
                    for j in range(i+1, vis.nant):
                        term1, term2 = self.separate_terms(vis.g_bar[time, freq, i], 
                                                      vis.g_bar[time, freq, j], 
                                                      vis.x[time, freq, i], 
                                                      vis.x[time, freq, j])

                        # Put them in the right place in the bigger matrix
                        proj[k*2, k*2] = term1
                        proj[k*2, k*2+1] = -term2
                        proj[k*2+1, k*2] = term2
                        proj[k*2+1, k*2+1] = term1

                        #proj_dok[k*2, k*2] = term1
                        #proj_dok[k*2, k*2+1] = -term2
                        #proj_dok[k*2+1, k*2] = term2
                        #proj_dok[k*2+1, k*2+1] = term1

                        k += 1

                # Append this block to a list
                mats.append(scipy.sparse.coo_matrix(proj))

        # Construct block diagonal sparse array from the projection operator of each antenna pair
        x = scipy.sparse.block_diag(mats, format='coo').toarray()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] != 0: print(i, j, x[i, j])
                
        exit()
        return scipy.sparse.block_diag(mats, format='coo')
    
    

    def generate_m_proj_sparse1(self, vis):
        """
        Generate the projection operator from gain parameter vectors 
        to visibilities as a CSR sparse array.

        This is a block diagonal array. Each block is for a given time and 
        frequency. Within each block, the projection operator is constructed 
        using the appropriate values of g_bar etc. for each antenna pair.
        """
        

        def make_index_vis(ti, fi, vi):
            """ Flat index , taking into account the re/im split
            vi is the index of the visibility value after splitting,
            it is not the index of a baseline. 
            """
            return ti*(nfreq*nvis*2)+fi*nvis*2+vi

        def insert(ti, fi, r, c, val, ind):
            rows[ind] = make_index_vis(ti, fi, r)
            cols[ind] = make_index_vis(ti, fi, c)
            data[ind] = val

                
        ntime = vis.g_bar.shape[0]
        nfreq = vis.g_bar.shape[1]
        nant = vis.g_bar.shape[2]
        nvis = vis.V_model.shape[2]

        nrows = ntime*nfreq*nvis*2
        ncols = nrows
        
        nvalues = ntime*nfreq*nvis*4
        value_index = 0

        # Generate the projection operator 
        rows = np.zeros(nvalues)
        cols = np.zeros(nvalues)
        data = np.zeros(nvalues)

        # Loop over times and frequencies
        for time in range(vis.ntime):
            for freq in range(vis.nfreq):

                bl = 0
                for ant1 in range(vis.nant):
                    for ant2 in range(ant1+1, vis.nant):
                        term1, term2 = self.separate_terms(vis.g_bar[time, freq, ant1], 
                                                      vis.g_bar[time, freq, ant2], 
                                                      vis.x[time, freq, ant1], 
                                                      vis.x[time, freq, ant2])

                        # Put them in the right place 
                        insert(time, freq, bl*2, bl*2, term1, value_index); value_index+=1
                        insert(time, freq, bl*2, bl*2+1, -term2, value_index); value_index+=1
                        insert(time, freq, bl*2+1, bl*2, term2, value_index); value_index+=1
                        insert(time, freq, bl*2+1, bl*2+1, term1, value_index); value_index+=1
                        
                        bl += 1


        for i in range(len(data)):
            print(rows[i], cols[i], data[i])
        exit()
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(nrows, ncols))


    def new_model_distribution(self, v):
        # The x values have been updated so get a new distribution.
        # If the x value has not been changed and C is set to None
        # and N = I then the mean of the distribution will be the 
        # v.V_model

        A = self.generate_m_proj(v)   # Square matrix of shape nvis*2*ntime*nfreq
        bm = BlockMatrix()
        bm.add(v.model_projection, replicate=v.ntime*v.nfreq)
        redundant_projector = bm.assemble()   # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime
        A = np.dot(A, redundant_projector)    # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime

    
        bm = BlockMatrix()
        for time in range(v.ntime):
            for freq in range(v.nfreq):
                bm.add((split_re_im(v.obs_variance[time][freq])))
        N = bm.assemble()
        
        bm = BlockMatrix()
        for time in range(v.ntime):
            for freq in range(v.nfreq):
                bm.add(self.Cv_diag)
        Cv = bm.assemble()
        
        
        V_mean = split_re_im(np.ravel(self.V_mean))
        d = split_re_im(np.ravel(v.V_obs))

        N_inv = np.linalg.inv(N)
        Cv_inv = np.linalg.inv(Cv)
        

        # Only want x values to go +/-10% 
        # Fiddle with the prior widths
        # Plot the distributions mathematically
        # Focus day in the office working on this
        # Equation 17
        # Add noise

        term1 = np.dot(A.T, np.dot(N_inv, A))
        dist_covariance = np.linalg.inv(term1+Cv_inv)
        term2 = np.dot(A.T, np.dot(N_inv, d))+np.dot(Cv_inv, V_mean)
        dist_mean = np.dot(dist_covariance, term2)


        """
        print("-------------", dist_mean, np.diag(dist_covariance))

        # Equation 13 far right

        sigma_1 = np.linalg.inv(np.dot(A.T, np.dot(np.linalg.inv(N), A)))
        mu_1 = np.dot(np.linalg.inv(A), d)
        sigma_2 = C
        mu_2 = V
        dist_mean = np.dot(np.dot(sigma_2, np.linalg.inv(sigma_1+sigma_2)), mu_1)+np.dot(np.dot(sigma_1, np.linalg.inv(sigma_1+sigma_2)), mu_2)
        dist_covariance = np.dot(sigma_1, np.dot(np.linalg.inv(sigma_1+sigma_2), sigma_2))
        print("------------", dist_mean, np.diag(dist_covariance))

        # Equation 13 right
        dist_covariance = np.dot(sigma_1, np.dot(np.linalg.inv(sigma_1+sigma_2), sigma_2))
        dist_mean = np.dot(dist_covariance, np.dot(np.linalg.inv(sigma_1), mu_1)+np.dot(np.linalg.inv(sigma_2), mu_2))
        print("-------------", dist_mean, np.diag(dist_covariance))
        exit()
        """

        return dist_mean, dist_covariance
    
    def distribute_redundant_V_samples(self):
        bm = BlockMatrix()
        bm.add(self.vis_redcal.model_projection, replicate=self.vis_redcal.ntime*self.vis_redcal.nfreq)
        redundant_projector = bm.assemble()   # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime

        
        V = np.matmul(redundant_projector, split_re_im(self.samples["V"].reshape(self.samples["V"].shape[0], -1)).T)
        V = unsplit_re_im(V.T)
        V = V.reshape(V.shape[0], self.vis_redcal.ntime, self.vis_redcal.nfreq, -1)
        
        return V
    
    def v_samp_V_unredundant(self):
        bm = BlockMatrix()
        bm.add(self.vis_redcal.model_projection, replicate=self.vis_redcal.ntime*self.vis_redcal.nfreq)
        redundant_projector = bm.assemble()   # Non square matrix of shape nvis*2*ntime*nfreq x nredundant_vis*2*nreq*ntime
        
        V = np.dot(redundant_projector, np.ravel(split_re_im(self.vis_sampled.V_model)))
        V = unsplit_re_im(V)
        V = np.reshape(V, (self.vis_redcal.ntime, self.vis_redcal.nfreq, -1))
        return V
    
    def bests(self, parameter, time=None, freq=None, method="mean", measure="rms"):
        """
        parameter: x or g
        measure is only used for method="ml"
        """
        
        def peak(a):
            # Get the peak 
            hist, bin_edges = np.histogram(a, bins=len(a)//10)
            bins = (bin_edges[1:]+bin_edges[:-1])/2
            return bins[np.argmax(hist)]

        assert parameter ==  "x" or parameter == "g", "Calculation of best sample must use x or g"

        best_dict = {}
        
        samples = { parameter: self.select_samples_by_time_freq(parameter, time, freq),
                       "V": self.select_samples_by_time_freq("V", time, freq)
                      }
        if method in [ "mean", "peak", "median" ]:
            for p in [ parameter, "V" ]:
                shape = samples[p].shape
                data = split_re_im(samples[p].reshape(shape[0], -1))
                if method == "mean":
                    result = np.mean(data, axis=0)
                elif method == "median":
                    result = np.median(data, axis=0)
                else:
                    result = np.array([ peak(data[:, i]) for i in range(data.shape[1]) ])
                best_dict[p] = unsplit_re_im(result).reshape(shape[1:])   

        elif method == "ml":

            vv = copy.deepcopy(self.vis_redcal)
            if parameter == "g": vv.x.fill(0)
            
            if measure == "likelihood":
                best = -1e39            # Want biggest
            else: best = 1e39           # Want smallest
            where_best = 0
            for i in range(samples["x"].shape[0]):   
                if parameter == "x":
                    vv.x = samples["x"][i]    
                else:
                    vv.g = samples["g"][i]  

                vv.V_model = samples["V"][i]  
                if measure == "likelihood":
                    lh = vv.get_unnormalized_likelihood(over_all=True)
                    if lh > best:
                        where_best = i
                        best = lh
                else:
                    rms = vv.get_rms()
                    if rms < best:
                        where_best = i
                        best = rms
                    
            # Return best values but this doesn't alter vis_sampled
            best_dict[parameter] = samples[parameter][where_best]
            best_dict["V"] = samples["V"][where_best]
        else:
            raise ValueError("Invalid method")

        return best_dict
    
    
    def fix_degeneracies(self):
        """
        Use the true (input) gains to fix the degeneracy directions in a set of 
        redundantly-calibrated gain solutions. This replaces the absolute 
        calibration that would normally be applied to a real dataset in order to 
        fix the degeneracies.

        Note that this step should only be using the true gains to fix the 
        degeneracies, and shouldn't add any more information beyond that.

        N.B. This is just a convenience function for calling the 
        remove_degen_gains() method of the redcal.RedundantCalibrator class. It 
        also assumes that only the 'ee' polarization will be used.

        Parameters
        ----------
        red_gains : dict of array_like
            Dict containing 2D array of complex gain solutions for each antenna 
            (and polarization).
            This is cal['g_omnical']

        true_gains : dict
            Dictionary of true (input) gains as a function of frequency. 
            Expected format: 
                key = antenna number (int)
                value = 1D numpy array of shape (Nfreqs,)
            This is the g_new.calfits file.


        Returns
        -------
        new_gains : dict
            Dictionary with the same items as red_gains, but where the degeneracies 
            have been fixed in the gain solutions.

        uvc : UVCal, optional
            If outfile is specified, also returns a UVCal object containing the 
            updated gain solutions.
        """
        try: hera_cal.__version__
        except: import hera_cal as hc
            
        def un_key(gains):
            stripped = np.empty((gains[(0, "Jee")].shape[0], gains[(0, "Jee")].shape[1], len(gains.keys())), 
                                dtype=type(gains[(0, "Jee")][0, 0]))
            for i, key in enumerate(gains):
                ant = key[0]
                stripped[:, : ,ant] = gains[key]
            return stripped
        
        def fix(cal, gains, gains_dict):
                # Fix degeneracies on "gains"

            for i in range(self.vis_redcal.nant):
                gains_dict[(i, "Jee")] = gains[:, :, i]
                
            new_gains = RedCal.remove_degen_gains(gains_dict,             # Dict containing 2D array of complex gain solutions for each antenna 
                                              degen_gains=true_gains, 
                                              mode='complex')
            return un_key(new_gains)
        
        assert len(self.file_root) > 0, "This is not a non-redundant sim. Can't fix degeneracies."

        print("Fixing degeneracies")
        
        if self.gain_degeneracies_fixed:
            print("Degeneracies already fixed. Nothing to do.")
            return
        
        # Need the redundant groups in the right format, antenna pairs and ee pol
        # Redundancy might have been removed but that's ok because each bl in a group still.
        reds = []
        for rg in self.vis_redcal.redundant_groups:
            new_rg = []
            for bl in rg:
                ants = self.vis_redcal.bl_to_ants[bl]
                ants = (ants[0], ants[1], "ee")
                new_rg.append(ants)
            reds.append(new_rg)

        """
        # This is what would do if using g_omnical
        # Need g_omnical
        fname = self.file_root+"_g_cal_dict.npz"
        red_gains = hkl.load(fname)["g_omnical"]
        # Keys are like (0, 'Jee'). For each key is an array (ntimes, nfreqs)
        # Have to select out the time freq of this object.
        assert len(red_gains.keys()) == self.nant
        for key in red_gains:
            red_gains[key] = red_gains[key][self.time:self.time+1, self.freq:self.freq+1]
        """
        
        time_range = self.time_range
        if time_range is None:
            time_range = ( 0, self.vis_redcal.ntime )
        freq_range = self.freq_range
        if freq_range is None:
            freq_range = ( 0, self.vis_redcal.nfreq )

           
        true_gains, _ = hc.io.load_cal(self.vis_redcal.file_root+"_g_new.calfits")
        # Same layout as red_gains
        assert len(true_gains.keys()) == self.vis_redcal.nant
        for key in true_gains:
            true_gains[key] = true_gains[key][time_range[0]:time_range[1], freq_range[0]:freq_range[1]]
        
        # Create calibrator and dict for the work
        RedCal = hc.redcal.RedundantCalibrator(reds)
        gains_dict = {}             # this is just used for temp space
        
        # We want to find x_new so that fix(g_bar(1+x)) = fix(g_bar)(1+x_new)
        #   x_new = fix(g_bar(1+x))/fix(g_bar) - 1
        
        # Fix all the samples
        
        fixed_g_bar = fix(RedCal, self.vis_redcal.g_bar.astype(np.complex128), gains_dict)
        
        if self.samples is not None:
            # Now fix the sampled gains and adjust the x values
            for i in range(self.samples["g"].shape[0]):

                self.samples["g"][i] = fix(RedCal, self.samples["g"][i], gains_dict)     # fix(g_bar(1+x))

                # Recalculate x
                self.samples["x"][i] = self.samples["g"][i]/fixed_g_bar - 1
  
        # Fix redcal gains
        fix_g_bar_1_plus_x = fix(RedCal, self.vis_redcal.get_antenna_gains().astype(np.complex128), gains_dict)
        self.vis_redcal.x = fix_g_bar_1_plus_x/fixed_g_bar - 1  # Update x
        self.vis_redcal.g_bar = fixed_g_bar

        if self.samples is not None:
            # Fix the x best sample
            fix_g_bar_1_plus_x = fix(RedCal, self.vis_sampled.get_antenna_gains().astype(np.complex128), gains_dict)
            self.vis_sampled.x = fix_g_bar_1_plus_x/fixed_g_bar - 1  # Update x
            self.vis_sampled.g_bar = fixed_g_bar

        self.gain_degeneracies_fixed = True
    


if __name__ == "__main__":
    from resource import getrusage, RUSAGE_SELF
    from s_manager import SManager
    import matplotlib.pyplot as plt
    import os, time, hickle, cProfile
    
    file_root = "/data/scratch/apw737/catall_nobright/viscatBC"
    sampler = Sampler(seed=99, niter=100, burn_in=10, best_type="mean", random_the_long_way=True, use_conj_grad=True, report_every=1000)
    sampler.load_nr_sim(file_root, remove_redundancy=False) 
    
    print(np.mean(sampler.vis_redcal.chi2["Jee"]))
    print(sampler.vis_redcal.get_chi2(over_all=True))
    exit()
    #sampler.load_sim(20, 40, 40)

    # Fourier mode setup for S
    dc = lambda x, y: 1 if x==0 and y == 0 else 0
    gauss = lambda x, y: np.exp(-0.5*(x**2+y**2)/.005)
    random = lambda x, y: np.random.random(size=1)

    sm = SManager(sampler.vis_redcal.ntime, sampler.vis_redcal.nfreq, sampler.vis_redcal.nant)
    sm.generate_S(gauss, modes=2, ignore_threshold=0, zoom_from=(64, 64), scale=2)    # Contains all times/freqs

    # V prior
    V_mean = sampler.vis_redcal.V_model
    Cv_diag = np.full(V_mean.shape[2]*2, 2)

    sampler.set_S_and_V_prior(sm, V_mean, Cv_diag)
    
    start = time.time()
    #cProfile.run("sampler.run()", filename="sampler.prof", sort="cumulative")
    sampler.run()
    sampler.fix_degeneracies()
    exit()


    print("Run time:", time.time()-start)
    
    dirname = "/scratch2/users/hgarsden/sampler_"+file_root.split("/")[-1]+"_"+str(os.getpid()) 
    try:
        os.mkdir(dirname)
    except: pass
    np.savez_compressed(dirname+"/"+"samples", x=sampler.samples["x"], g=sampler.samples["g"], V=sampler.samples["V"])
    sampler.fops = sampler.S = sampler.samples["x"] = sampler.samples["g"] = sampler.samples["V"] =None
    hickle.dump(sampler, dirname+"/sampler.hkl", mode='w', compression='gzip')
    print("Wrote", dirname)

    exit()

 
    x = sampler.x_random_draw(sampler.vis_redcal)
    x = sampler.reform_x_from_samples(x, sampler.vis_redcal.x.shape) 
    x = np.moveaxis(x, 2, 0)

    plt.clf()
    plt.plot(x[0, :, 4].real)
    plt.savefig("x")
    
    usage = getrusage(RUSAGE_SELF)
    print("SIM", usage.ru_maxrss/1000.0/1000)      # Usage in GB
    
    exit()
  
