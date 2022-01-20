import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vis_creator import VisSim, VisCal, VisTrue
from gls import gls_solve, generate_proj
from calcs import split_re_im, unsplit_re_im, BlockMatrix, remove_x_im, restore_x_im, get_cov_matrix
import hera_cal as hc
import corner
import copy
import numpy as np
import scipy.linalg, scipy.sparse


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
    
    def __init__(self, niter=1000, burn_in=10, seed=None, random_the_long_way=False, use_conj_grad=True, best_type="mean"):


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
        self.best_type = best_type
        
        self.gain_degeneracies_fixed = False
    
    def load_nr_sim(self, file_root, time_range=None, freq_range=None, remove_redundancy=False, initial_solve_for_x=False):
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

        Returns
        -------
        vis : ndarray of complex
            Array of visibilities at each LST and frequency appropriate
            for the given sky temperature model, beam size model, and
            baseline vector. Returned in units of Jy with shape
            (lsts.size, freqs.size).
        """

        print("Loading NR sim from", file_root)
        self.vis_redcal = VisCal(file_root, time_range=time_range, freq_range=freq_range, remove_redundancy=remove_redundancy) 
        self.vis_true = VisTrue(file_root, time_range=time_range, freq_range=freq_range)

        if initial_solve_for_x:
            self.vis_redcal.x = self.vis_redcal.initial_vals.x = gls_solve(self.vis_redcal)
            
        # Indexes into the original simulated data
        self.time_range = time_range
        self.freq_range = freq_range
        
        self.file_root = file_root
            
            
    def load_sim(self, nant, ntime=1, nfreq=1, initial_solve_for_x=False, **kwargs):

        self.vis_redcal = VisSim(nant, ntime=ntime, nfreq=nfreq, **kwargs)
        self.vis_true = self.vis_redcal
        if initial_solve_for_x:
            self.vis_redcal.x = self.vis_redcal.initial_vals.x = gls_solve(self.vis_redcal)

        self.file_root = ""
            
    def set_S_and_V_prior(self, S_diag, V_mean, Cv_diag):
        self.S = get_cov_matrix(self.vis_redcal.ntime, self.vis_redcal.nfreq, self.vis_redcal.nant, S_diag)
        self.V_mean = V_mean
        self.Cv = get_cov_matrix(self.vis_redcal.ntime, self.vis_redcal.nfreq, self.vis_redcal.nant, Cv_diag)
        
    def nant(self):
        return self.vis_redcal.nant
    
    def nvis(self):
        return self.vis_redcal.nvis
    
                   
    def run(self):
        if not hasattr(self,"vis_true"):
            raise RuntimeError("No sim loaded. Can't sample.")
            
        print("Running sampling")
        sampled_x = np.zeros((self.niter, self.vis_redcal.ntime, self.vis_redcal.nfreq, self.vis_redcal.nant),
                            dtype=type(self.vis_redcal.x[0, 0, 0]))   
                                                                # -1 because there'll be a missing imaginary value
        sampled_V = np.zeros((self.niter, self.vis_redcal.ntime, self.vis_redcal.nfreq, len(self.vis_redcal.redundant_groups)),
                             dtype=type(self.vis_redcal.V_model[0, 0, 0]))
 
        v_x_sampling = copy.deepcopy(self.vis_redcal)      
        v_model_sampling = copy.deepcopy(self.vis_redcal) 

        new_x = v_model_sampling.x         # Initialize
        
        # Take num samples
        for i in range(self.niter):

            # Use the sampled x to change the model sampling distribution, and take a sample
            v_model_sampling.x = new_x
            if self.random_the_long_way:
                sample = self.V_random_draw(v_model_sampling)
            else:
                v_dist_mean, v_dist_covariance = self.new_model_distribution(v_model_sampling)
                sample = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)
            
            new_model = unsplit_re_im(sample)
            sampled_V[i] = new_model.reshape(self.vis_redcal.V_model.shape)

            # Use the sampled model to change the x sampling distribution, and take a sample
            v_x_sampling.V_model = sampled_V[i]
            if self.random_the_long_way:
                sample = self.x_random_draw(v_x_sampling)
            else:
                x_dist_mean, x_dist_covariance = self.new_x_distribution(v_x_sampling)
                sample = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  
            
            new_x = self.reform_x_from_samples(sample, self.vis_redcal.x.shape) 
            sampled_x[i] = new_x
            
        sampled_x = sampled_x[(self.niter*self.burn_in)//100:]
        sampled_V = sampled_V[(self.niter*self.burn_in)//100:]
        sampled_gains = np.zeros_like(sampled_x)
        for i in range(sampled_gains.shape[0]):
            sampled_gains[i] = self.vis_redcal.g_bar*(1+sampled_x[i])
        
        self.samples = {
            "x" : sampled_x,
            "g" : sampled_gains,
            "V" : sampled_V
        }
        
        # Create an object containing the best fit
        self.vis_sampled = copy.deepcopy(self.vis_redcal)    
        best_vals = self.bests(method=self.best_type)
        self.vis_sampled.x = best_vals["x"]
        self.vis_sampled.V_model = best_vals["V"]
    
    def select_samples_by_time_freq(self, what, time=None, freq=None):
        assert what in [ "x", "g", "V" ], "Invalid sample specification"

        if time is None and freq is not None:
            assert False, "Both a time and frequency must be specified"
            samples = self.samples[what][:, :, freq:freq+1, :]
        elif time is not None and freq is None:
            assert False, "Both a time and frequency must be specified"
            samples = self.samples[what][:, time:time+1, :, :]
        elif time is not None and freq is not None:
            samples = self.samples[what][:, time:time+1, freq:freq+1, :]
        else:
            samples = self.samples[what]

        return samples
            
    def select_values_by_time_freq(self, values, time=None, freq=None):

        if time is None and freq is not None:
            assert False, "Both a time and frequency must be specified"
            v = values[:, freq:freq+1, :]
        elif time is not None and freq is None:
            assert False, "Both a time and frequency must be specified"
            v = values[time:time+1, :, :]
        elif time is not None and freq is not None:
            v = values[time:time+1, freq:freq+1, :]
        else:
            v = values
            
        return v
            
    def remove_unsampled_x(self, x):
        samples_x = split_re_im(x)    
        samples_x = np.delete(samples_x, samples_x.shape[3]-1, axis=3)     # Remove unsampled x
        samples_x = samples_x.reshape(samples_x.shape[0], -1)
            
        return samples_x
            
    def reform_x_from_samples(self, sampled_x, shape):
        x = sampled_x.reshape(shape[0], shape[1], shape[2]*2-1)
        x = restore_x_im(x)
        return unsplit_re_im(x)

    
    def plot_marginals(self, parameter, cols, time=None, freq=None, which=[ "True", "Redcal", "Sampled" ]):
        def plot_hist(a, fname, label, sigma_prior, other_vals, index):
            hist, bin_edges = np.histogram(a, bins=len(a)//50)
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

                plot_hist(samples_V[:, i], part+"_V_"+str(i//2), part+"(V_"+str(i//2)+")", None, other_vals, i+1)
        else:
            raise ValueError("Invalid spec for plot_marginals")
            
        plt.tight_layout()
        
        
    def plot_corner(self, parameters, time=None, freq=None, threshold=0.0, xgs=None, Vs=None):
        assert parameters == ["x", "V"] or parameters == ["g", "v"], "corner plot needs x,V or g,V"
        assert threshold >= 0
        assert (threshold == 0.0 and xgs is None and Vs is None) or \
                (threshold > 0.0 and xgs is None and Vs is None) or \
                (threshold == 0.0 and xgs is not None and Vs is not None), \
                "Must have: threshold==0 and xgs=list and Vgs==list OR threshold>0 and xgs=None and Vgs==None"+ \
                "\nOR threshold==0 and xgs=None and Vgs==None"
                
        
        print("Plot corner")
        
        data_packet = self.assemble_data(parameters, time, freq)       # Puts both together flat

        part = lambda i : "re" if i%2==0 else "im"
       
        name = "x" if "x" in parameters else "g"
        labels = [ r"$"+part(i)+"("+name+"_"+str(i//2)+")$" for i in range(data_packet["x_or_g_len"]) ]+  \
                    [ r"$"+part(i)+"(V_"+str(i//2)+")$" for i in range(data_packet["V_len"]) ]
        
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
            
        if xgs is not None and Vs is not None:
            Vs = [ v+data_packet["x_or_g_len"] for v in Vs ]
            
            to_delete = np.arange(data_packet["data"].shape[0])[np.isin(np.arange(data_packet["data"].shape[0]), xgs+Vs, invert=True)]
            labels = [ labels[i] for i in range(len(labels)) if i not in to_delete ]
            num_x_left = data_packet["x_or_g_len"]-np.where(to_delete<data_packet["x_or_g_len"])[0].size
         
            data_packet["data"] = np.delete(data_packet["data"], to_delete, axis=1)
            data_packet["x_or_g_len"] = num_x_left
            data_packet["V_len"] = data_packet["data"].shape[1]-num_x_left
            
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

        #return figure
        
    def plot_trace(self, parameter, time=None, freq=None, index=None):
        assert isinstance(parameter, str), "trace parameter must be string"
        assert parameter in [ "x", "g", "V" ], "Unknown parameter: "+parameter
        
        print("Plot trace")
        
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
                    [ part(i)+"(V_"+str(i//2)+")" for i in range(data_packet["V_len"]) ]
        
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
            
    def plot_covcorr(self, parameters, time=None, freq=None, stat="corr", threshold=0.0):
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
        print("Matrix size", m.shape)

        np.fill_diagonal(m, 0)
        m[np.triu_indices(m.shape[0])] = 0
        if threshold > 0.0: m[np.logical_and(m>-threshold,m<threshold)] = 0

        plt.figure()
        ax = plt.gca()
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
        

    def plot_results(self, time=None, freq=None): 
        
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
        plt.plot(np.abs(v_true[order]), 
                 np.abs(v_sampled[order]), "b", linewidth=0.6,  label="Sampled")
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
        plt.plot(np.angle(v_true.astype(np.complex64)[order]), 
                 np.angle(v_sampled.astype(np.complex64)[order]), "b", linewidth=0.6,  label="Sampled")
        plt.legend()
        plt.xlabel("V_true phase")
        plt.ylabel("Phase")
        plt.title("Calibrated visibilities (Phase)")
        plt.tight_layout()
        
    def plot_gains(self, time=None, freq=None, sigma=3):
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
        g_sampled = self.select_values_by_time_freq(self.vis_sampled.get_antenna_gains(), time, freq)      
        
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
        plt.plot(range(g_true.size), np.angle(g_true), color="green", linewidth=0.6, label="g_true")

        # 2. The redcal gains as they are given to us by redcal.
        plt.plot(range(g_true.size), np.angle(g_redcal.astype(np.complex64)), "r", linewidth=0.6, label="g_redcal")

        # 3. The sampled gains, actually x is sampled. 
        plt.plot(range(g_true.size), np.angle(g_sampled.astype(np.complex64)), "b", linewidth=0.6, label="g_sampled")
        
        # Error bars
        assert error_phase.shape[0] == g_true.size
        for i in range(error_phase.shape[0]):
            plt.plot([i, i], [ error_phase[i][0], error_phase[i][1] ], "lightblue")
        plt.legend()
        plt.title("Gain phases")
        plt.xlabel("Antenna")
        plt.ylabel("Phase (rad)")
        plt.tight_layout()

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
        mean = np.zeros(size)
        cov = np.eye(size)
        return np.random.multivariate_normal(mean, cov)
    
    def sqrtm(self, m):
        m = scipy.linalg.sqrtm(m)
        
        # np.iscomplex will return False even if the number is complex but the imag
        # part is 0
    
        if np.iscomplex(m[0, 0]):
            assert not np.any(m.imag), "Square root of matrix is complex - cannot continue"
            
        return m.real
    
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

                           
    def random_draw(self, first_term, A, S, N):
        N_inv = np.linalg.inv(N)
        S_inv = np.linalg.inv(S) 
        S_sqrt = self.sqrtm(S)
        A_N_A = np.dot(A.T, np.dot(N_inv, A))
        
        
        rhs = np.dot(S_sqrt, first_term)
        rhs += np.dot(S_sqrt, np.dot(A.T, np.dot(self.sqrtm(N_inv), self.standard_random_draw(A.shape[0]))))
        rhs += self.standard_random_draw(A.shape[1])
        
        bracket_term = np.eye(S.shape[0])
        bracket_term += np.dot(S_sqrt, np.dot(A_N_A, S_sqrt))
        
        if self.use_conj_grad:
            x, info = scipy.sparse.linalg.cg(bracket_term, rhs)
            assert info == 0
        else: x = np.dot(np.linalg.inv(bracket_term), rhs)
        
        return np.dot(S_sqrt, x)

    
    def x_random_draw(self, v):
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
                bm.add((self.S))
        S = bm.assemble()
        
        return self.random_draw(np.dot(A.T, np.dot(np.linalg.inv(N), d)), A, S, N)
        
    
    def V_random_draw(self, v):
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
                bm.add(self.Cv)
        Cv = bm.assemble()
        
        V_mean = split_re_im(np.ravel(self.V_mean))
        d = split_re_im(np.ravel(v.V_obs))
                                
        return self.random_draw(np.dot(A.T, np.dot(np.linalg.inv(N), d))+np.dot(np.linalg.inv(Cv), V_mean), A, Cv, N)


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
        
        S = self.S

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
        
        Cv = self.Cv
        
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
    
    def bests(self, parameters=["x", "V"], method="mean"):
        def peak(a):
            # Get the peak 
            hist, bin_edges = np.histogram(a, bins=len(a)//10)
            bins = (bin_edges[1:]+bin_edges[:-1])/2
            return bins[np.argmax(hist)]

        assert parameters == ["x", "V"] or parameters == ["g", "V"], "Calculation of best sample must use x,V or g,V"

        best_dict = {}
        
        if method == "mean":
            for p in parameters:
                data = split_re_im(self.samples[p])
                means = np.mean(data, axis=0)
                best_dict[p] = unsplit_re_im(means)

        elif method == "peak":
             for p in parameters:
                shape = self.samples[p].shape
                data = split_re_im(self.samples[p].reshape((shape[0], -1)))
                peaks = np.array([ peak(data[:, i]) for i in range(data.shape[1]) ])
                best_dict[p] = unsplit_re_im(peaks).reshape(shape[1:])
                
        elif method == "ml":

            vv = copy.deepcopy(self.vis_redcal)
            assert np.sum(np.abs(vv.x)) == 0, "Redcal object contains x values"
                    
            best = -1e39
            where_best = 0
            for i in range(self.samples["x"].shape[0]):   
                if "x" in parameters:
                    vv.x = self.samples["x"][i]    
                else:
                    vv.g = self.samples["g"][i]  
                V_model = self.samples["V"][i]  
                lh = vv.get_unnormalized_likelihood(over_all=True)
                if lh > best:
                    where_best = i
                    best = lh

            if "x" in parameters:
                best_dict["x"] = self.samples["x"][where_best]

                # Generate gains from best x
                vv.x = self.samples["x"][where_best]
                best_dict["g"] = vv.get_antenna_gains()
            else:
                best_dict["g"] = self.samples["g"][where_best]
            best_dict["V"] = self.samples["V"][where_best]
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
        
        def un_key(gains):
            stripped = np.empty((gains[(0, "Jee")].shape[0], gains[(0, "Jee")].shape[1], len(gains.keys())), 
                                dtype=type(gains[(0, "Jee")][0, 0]))
            for i, key in enumerate(gains):
                ant = key[0]
                stripped[:, : ,ant] = gains[key]
            return stripped
        
        def fix(cal, gains, gains_dict):
                # Fix degeneracies on gains

            for i in range(self.vis_redcal.nant):
                gains_dict[(i, "Jee")] = gains[:, :, i]
                
            new_gains = RedCal.remove_degen_gains(gains_dict, 
                                              degen_gains=true_gains, 
                                              mode='complex')
            return un_key(new_gains)
        
        assert len(self.file_root) > 0, "This is not a non-redundant sim. Can't fix degeneracies."
        
        if self.gain_degeneracies_fixed:
            print("Degeneracies already fixed. Nothing to do.")
            return
        
        # Need the redundant groups in the right format, antenna pairs and ee pol
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
        
        # Fix all the samples
        
        # Now fix the sampled gains and adjust the x values
        for i in range(self.samples["g"].shape[0]):
            samples_orig_x = np.copy(self.samples["x"][i])
            samples_orig_g = np.copy(self.samples["g"][i])
            self.samples["g"][i] = fix(RedCal, samples_orig_g, gains_dict)
            
            # Recalculate x
            self.samples["x"][i] = samples_orig_g*(1+samples_orig_x)/self.samples["g"][i] - 1
  
        # Fix redcal gains
    
        self.vis_redcal.g_bar = fix(RedCal, self.vis_sampled.get_antenna_gains().astype(np.complex64), gains_dict)
        self.vis_redcal.x.fill(0)            # should be 0 unless initial_solve_for_x

        # Fix the x best sample
        
        orig_g = self.vis_sampled.g_bar
        orig_x = self.vis_sampled.x

        self.vis_sampled.g_bar = self.vis_redcal.g_bar      # Update gain
        self.vis_sampled.x = (orig_g*(1+orig_x))/self.vis_sampled.g_bar - 1  # Update x
                          
        self.gain_degeneracies_fixed = True
    


if __name__ == "__main__":
    from resource import getrusage, RUSAGE_SELF

    sampler = Sampler(seed=99, niter=1000, burn_in=10, best_type="mean", random_the_long_way=True)
    sampler.load_sim(4, ntime=1, nfreq=1, x_sigma=0)
    print("Likelihood before run", sampler.vis_true.get_unnormalized_likelihood(unity_N=True))   
    S = np.eye(sampler.nant()*2-1)*0.01
    V_mean = sampler.vis_redcal.V_model
    Cv = np.eye(V_mean.shape[2]*2)
    sampler.set_S_and_V_prior(S, V_mean, Cv)

    sampler.run()
    print("Likelihood after run", sampler.vis_sampled.get_unnormalized_likelihood(unity_N=True))   
    
    usage = getrusage(RUSAGE_SELF)
    print("SIM", usage.ru_maxrss/1000.0/1000)      # Usage in GB
    
    exit()
    
    
    cv_factor = 1
    for i in range(10):
        print("cv", cv_factor, "===================================================")
        
        s_factor = 1
        
        for j in range(10):
            print("s", s_factor, "===================================================")
            
            sampler = Sampler(seed=99, niter=10000, burn_in=10, best_type="mean", random_the_long_way=False)
            sampler.load_nr_sim("/scratch3/users/hgarsden/catall/calibration_points/viscatBC", 
                            time_range=(0, 2), freq_range=(0, 2), remove_redundancy=False, initial_solve_for_x=False)  
            print("Likelihood before run", sampler.vis_true.get_unnormalized_likelihood(over_all=True, unity_N=True))   

            S = np.eye(sampler.nant()*2-1)*0.01*s_factor
            V_mean = sampler.vis_redcal.V_model #*np.random.rand(sampler.vis_redcal.V_model.shape[0],
                                                              #sampler.vis_redcal.V_model.shape[1],
                                                              #sampler.vis_redcal.V_model.shape[2])
            Cv = np.eye(V_mean.shape[2]*2)*cv_factor
            sampler.set_S_and_V_prior(S, V_mean, Cv)

            sampler.run()
            print("Likelihood after run", sampler.vis_sampled.get_unnormalized_likelihood(over_all=True, unity_N=True))   
            sampler.print_covcorr(["x", "V"], count_them=True, threshold=0.8)
            
            s_factor *= 1.3
            
            
        cv_factor *= 1.3
        
    
    
    #sampler.plot_marginals("x", 5, time=0, freq=0); plt.clf()
    #sampler.plot_trace("x", time=0, freq=0); plt.clf()
    #sampler.plot_covcorr(["x", "V"],  time=0, freq=None); plt.clf()
    #sampler.plot_sample_means(["x"],  time=0, freq=0); plt.clf()
    #sampler.plot_results(time=0, freq=0); plt.clf()
    #sampler.plot_gains(); plt.clf()
    #sampler.plot_corner(["x", "V"], time=0, freq=0, threshold=0.8)
