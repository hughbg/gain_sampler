import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from vis_creator import VisSim, VisCal, VisTrue
from gls import gls_solve, generate_proj, generate_proj1, reduce_dof, restore_x
from calcs import split_re_im, unsplit_re_im
import hera_cal as hc
import corner
import copy
import numpy as np
import scipy.linalg



class Sampler:
    
    def __init__(self, niter=1000, burn_in=10, seed=None, random_the_long_way=False, best_type="mean"):
        if seed is not None:
            np.random.seed(seed)
        self.niter = niter
        self.burn_in = burn_in
        self.random_the_long_way = random_the_long_way
        self.best_type = best_type
        self.gain_degeneracies_fixed = False
    
    def load_nr_sim(self, file_root, time=0, freq=0, remove_redundancy=False, initial_solve_for_x=False):
        print("Loading NR sim from", file_root)
        self.vis_redcal = VisCal(file_root, time=time, freq=freq, remove_redundancy=remove_redundancy) 
        self.vis_true = VisTrue(file_root, time=time, freq=freq)

        if initial_solve_for_x:
            self.vis_redcal.x = self.vis_redcal.initial_vals.x = gls_solve(self.vis_redcal)
            
        self.file_root = file_root
            
            
    def load_sim(self, nant, initial_solve_for_x=False, **kwargs):
        self.vis_redcal = VisSim(nant, **kwargs)
        self.vis_true = self.vis_redcal
        if initial_solve_for_x:
            self.vis_redcal.x = self.vis_redcal.initial_vals.x = gls_solve(self.vis_redcal)

        self.file_root = ""
            
    def set_S_and_V_prior(self, S, V_mean, Cv):
        self.S = S
        self.V_mean = V_mean
        self.Cv = Cv
        
    def nant(self):
        return self.vis_redcal.nant
    
    def nvis(self):
        return self.vis_redcal.nvis
                   
    def run(self):
        if not hasattr(self,"vis_true"):
            raise RuntimeError("No sim loaded. Can't sample.")
            
        print("Running sampling")
        sampled_x = np.zeros((self.niter, self.vis_redcal.nant*2-1))       # -1 because there'll be a missing imaginary value
        sampled_V = np.zeros((self.niter, self.V_mean.size*2))
 
        v_x_sampling = copy.deepcopy(self.vis_redcal)      
        v_model_sampling = copy.deepcopy(self.vis_redcal) 

        new_x = v_model_sampling.x         # Initialize
        
        # Take num samples
        for i in range(self.niter):
            # Use the sampled x to change the model sampling distribution, and take a sample
            v_model_sampling.x = new_x
            if self.random_the_long_way:
                sampled_V[i] = self.V_random_draw(v_model_sampling)
            else:
                v_dist_mean, v_dist_covariance = self.new_model_distribution(v_model_sampling)
                sampled_V[i] = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)
            
            new_model = unsplit_re_im(sampled_V[i])

            # Use the sampled model to change the x sampling distribution, and take a sample
            v_x_sampling.V_model = new_model
            if self.random_the_long_way:
                sampled_x[i] = self.x_random_draw1(v_x_sampling)
            else:
                x_dist_mean, x_dist_covariance = self.new_x_distribution(v_x_sampling)
                sampled_x[i] = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  
            
            new_x = unsplit_re_im(restore_x(sampled_x[i]))
                        
        sampled_x = sampled_x[(self.niter*self.burn_in)//100:]
        sampled_V = sampled_V[(self.niter*self.burn_in)//100:]
        sampled_gains = np.zeros((sampled_x.shape[0], sampled_x.shape[1]+1))
        for i in range(sampled_x.shape[0]):
            x = unsplit_re_im(restore_x(sampled_x[i]))
            sampled_gains[i] = split_re_im(self.vis_redcal.g_bar*(1+x))  
        
        self.samples = {
            "x" : sampled_x,
            "g" : sampled_gains,
            "V" : sampled_V
        }
        
        # Create an object containing the best fit
        self.vis_sampled = copy.deepcopy(self.vis_redcal)    
        best_vals = self.bests(method=self.best_type)
        self.vis_sampled.x = unsplit_re_im(restore_x(best_vals["x"]))
        self.vis_sampled.V_model = unsplit_re_im(best_vals["V"])  
    
    def plot_marginals(self, parameter, cols, which=[ "True", "Redcal", "Sampled" ]):
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
        assert len(which) > 0, "No results specified to plot"
        for w in which:
            assert w in [ "True", "Redcal", "Sampled" ], "Invalid results to plot: "+str(w)
        
        if parameter == "x":
            num_plots = self.samples["x"].shape[1]
            if num_plots%cols == 0: rows = num_plots//cols
            else: rows = num_plots//cols+1
            
            true_x = (split_re_im(self.vis_true.g_bar)/split_re_im(self.vis_redcal.g_bar)-1)
            redcal_x = split_re_im(self.vis_redcal.x)
            sampled_x = split_re_im(self.vis_sampled.x)
            for i in range(self.samples["x"].shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {}
                if "True" in which: other_vals["True"] = ( true_x[i], "green" )
                if "Redcal" in which: other_vals["Redcal"] = ( redcal_x[i], "red" )
                if "Sampled" in which: other_vals["Sampled"] = ( sampled_x[i], "blue" )

                plot_hist(self.samples["x"][:, i], part+"_x_"+str(i//2), part+"(x_"+str(i//2)+")", None, other_vals, i+1)
        elif parameter == "g":
            num_plots = self.samples["g"].shape[1]
            if num_plots%cols == 0: rows = num_plots//cols
            else: rows = num_plots//cols+1
            
            true_g = split_re_im(self.vis_true.g_bar)
            redcal_g = split_re_im(self.vis_redcal.g_bar)
            sampled_g = split_re_im(self.vis_sampled.get_antenna_gains())
            for i in range(self.samples["g"].shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {}
                if "True" in which: other_vals["True"] = ( true_g[i], "green" )
                if "Redcal" in which: other_vals["Redcal"] = ( redcal_g[i], "red" )
                if "Sampled" in which: other_vals["Sampled"] = ( sampled_g[i], "blue" )

                plot_hist(self.samples["g"][:, i], part+"_g_"+str(i//2), part+"(g_"+str(i//2)+")", None, other_vals, i+1)
                
        elif parameter == "V":
            num_plots = self.samples["V"].shape[1]
            if num_plots%cols == 0: rows = num_plots//cols
            else: rows = num_plots//cols+1
            
            true_V = split_re_im(self.vis_true.V_model)
            redcal_V = split_re_im(self.vis_redcal.V_model)
            sampled_V = split_re_im(self.vis_sampled.V_model)
            for i in range(self.samples["V"].shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {}
                if "True" in which: other_vals["True"] = ( true_V[i], "green" )
                if "Redcal" in which: other_vals["Redcal"] = ( redcal_V[i], "red" )
                if "Sampled" in which: other_vals["Sampled"] = ( sampled_V[i], "blue" )

                plot_hist(self.samples["V"][:, i], part+"_V_"+str(i//2), part+"(V_"+str(i//2)+")", None, other_vals, i+1)
        else:
            raise ValueError("Invalid spec for plot_marginals")
            
        plt.tight_layout()
        
        
    def plot_corner(self, parameters):
        assert len(parameters) == 2, "corner plot needs x,V or g,V"
        data_packet = self.assemble_data(parameters)       # Puts both together
        
        part = lambda i : "re" if i%2==0 else "im"
       
        name = "x" if "x" in parameters else "g"
        
        labels = [ r"$"+part(i)+"("+name+"_"+str(i//2)+")$" for i in range(data_packet["x_or_g_len"]) ]+  \
                    [ r"$"+part(i)+"(V_"+str(i//2)+")$" for i in range(self.samples["V"].shape[1]) ]

        figure = corner.corner(data_packet["data"], labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)
        
        axes = np.array(figure.axes).reshape((data_packet["data"].shape[1], data_packet["data"].shape[1]))
        for i in range(data_packet["data"].shape[1]):
            for j in range(data_packet["data"].shape[1]):
                ax = axes[i, j]
                if i < data_packet["x_or_g_len"]:
                    color = "blue"
                else:
                    if j < data_packet["x_or_g_len"]: color = "pink"
                    else:
                        color = "green"
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
        
    def plot_trace(self, parameter):
        assert isinstance(parameter, str), "trace parameter must be string"
        assert parameter in [ "x", "g", "V" ], "Unknown parameter: "+parameter
        
        data = self.samples[parameter]
            
        sample_range = np.arange((self.burn_in*self.niter)//100, self.niter, 1)
        for i in range(data.shape[1]):
            plt.plot(sample_range, data[:, i])
            
        plt.xlabel("Sample iteration")
        plt.ylabel(parameter)
        plt.title("Traces for "+parameter)
                 
    def print_covcorr(self, parameters, stat="corr", threshold=0.0, list_them=False):
        assert len(parameters) == 2, "covariance needs x,V or g,V"
        assert stat == "cov" or stat == "corr", "Specify cov or corr for matrix"
        
        part = lambda i : "re" if i%2==0 else "im"

        data_packet = self.assemble_data(parameters)
        m = np.cov(data_packet["data"], rowvar=False) if stat == "cov" else np.corrcoef(data_packet["data"], rowvar=False) 

        name = "x" if "x" in parameters else "g"
        labels = [ part(i)+"("+name+"_"+str(i//2)+")" for i in range(data_packet["x_or_g_len"]) ]+  \
                    [ part(i)+"(V_"+str(i//2)+")" for i in range(self.samples["V"].shape[1]) ]
        
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
            
    def plot_covcorr(self, parameters, stat="corr", threshold=0.0):
        assert len(parameters) == 2, "covariance needs x,V or g,V"
        assert stat == "cov" or stat == "corr", "Specify cov or corr for matrix"
        
        part = lambda i : "re" if i%2==0 else "im"
        
        if stat == "corr":
            print("Plotting correlation matrix")
        else:
            print("Plotting covariance matrix")

        data_packet = self.assemble_data(parameters)
        m = np.cov(data_packet["data"], rowvar=False) if stat == "cov" else np.corrcoef(data_packet["data"], rowvar=False) 
        
        np.fill_diagonal(m, 0)
        m[np.triu_indices(m.shape[0])] = 0
        m[np.logical_and(m>-threshold,m<threshold)] = 0
    
        plt.figure()
        ax = plt.gca()
        im = ax.matshow(m, cmap="RdBu")
        nx = data_packet["x_or_g_len"]
        nv = self.samples["V"].shape[1]
        param_tag = "x" if "x" in parameters else "g"
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
            
    def plot_sample_means(self, parameters):
        assert len(parameters) <= 3, "Too many parameters requested"
        for p in parameters:
            assert p in [ "x", "g", "V" ], "Invalid parameter: "+p
            
        for i, p in enumerate(parameters):
            data_packet = self.assemble_data([p])
            plt.subplot(len(parameters), 1, i+1)
            plt.plot(range(data_packet["data"].shape[1]), np.mean(data_packet["data"], axis=0), label="Means", linewidth=0.6)
            x=plt.xticks(range(0, data_packet["data"].shape[1], 2), range(0, data_packet["data"].shape[1], 2))
            plt.xlabel("Order")
            plt.legend()
            plt.title("Sample means \""+p+"\" (split real/imag) in order")

        
        plt.tight_layout()
        

    def plot_results(self): 
        plt.subplot(2, 1, 1)
        order = np.abs(self.vis_true.V_model).argsort()
        plt.plot(np.abs(self.vis_true.V_model)[order], 
                 np.abs(self.vis_true.V_model)[order], "k", linewidth=0.6,  label="1:1")
        plt.plot(np.abs(self.vis_true.V_model)[order], 
                 np.abs(self.vis_redcal.get_calibrated_visibilities())[order], "r", linewidth=0.6,  label="Redcal")
        plt.plot(np.abs(self.vis_true.V_model)[order], 
                 np.abs(self.vis_sampled.get_calibrated_visibilities())[order], "b", linewidth=0.6,  label="Sampled")
        plt.legend()
        plt.xlabel("V_true amplitude")
        plt.ylabel("Amplitude")
        plt.title("Calibrated visibilities (Amplitude)")
        
        plt.subplot(2, 1, 2)
        order = np.angle(self.vis_true.V_model).argsort()
        plt.plot(np.angle(self.vis_true.V_model.astype(np.complex64))[order], 
                 np.angle(self.vis_true.V_model.astype(np.complex64))[order], "k", linewidth=0.6,  label="1:1")
        plt.plot(np.angle(self.vis_true.V_model.astype(np.complex64))[order], 
                 np.angle(self.vis_redcal.get_calibrated_visibilities().astype(np.complex64))[order], "r", linewidth=0.6,  label="Redcal")
        plt.plot(np.angle(self.vis_true.V_model.astype(np.complex64))[order], 
                 np.angle(self.vis_sampled.get_calibrated_visibilities().astype(np.complex64))[order], "b", linewidth=0.6,  label="Sampled")
        plt.legend()
        plt.xlabel("V_true phase")
        plt.ylabel("Phase")
        plt.title("Calibrated visibilities (Phase)")
        plt.tight_layout()
        
    def plot_gains(self, sigma=3):
        def normalize_phases(phases):
            # Make sure phases are between -pi, pi
            phases = np.where(phases>3*np.pi/2, phases-2*np.pi, phases)
            phases = np.where(phases<-3*np.pi/2, phases+2*np.pi, phases)
            return phases
        
        def get_g_error_bars():
            SIG = 3
            BOTTOM = 0
            TOP = 1
            
            sampled_gains = unsplit_re_im(self.samples["g"])   # Note need to unsplit                               
            
            g_limits_amp = np.zeros((sampled_gains.shape[1], 2))    
            g_limits_phase = np.zeros((sampled_gains.shape[1], 2))
            
            # Get the error bars. For each g, get the range based on SIG sigma.
            for i in range(sampled_gains.shape[1]):       # Loop over antennas
                amps = np.abs(sampled_gains[:, i])
                m = np.mean(amps)
                s = np.std(amps)
                g_limits_amp[i, BOTTOM] = m-SIG*s
                g_limits_amp[i, TOP] = m+SIG*s
                
                phases = normalize_phases(np.angle(sampled_gains[:, i]))
                m = np.mean(phases)
                s = np.std(phases)
                g_limits_phase[i, BOTTOM] = m-SIG*s
                g_limits_phase[i, TOP] = m+SIG*s
                
                
            return g_limits_amp, g_limits_phase
        
        
        #g_bar = self.v_sampled.g_bar
        #print(get_x_error_bars())
        #exit()
        
        error_amp, error_phase = get_g_error_bars()
        
        plt.subplot(2, 1, 1)

        # 1. V true. The gains are 1? Can't remember why
        plt.plot(range(self.vis_redcal.nant), np.abs(self.vis_true.get_antenna_gains()), color="green", linewidth=0.6, label="g_true")

        # 2. The redcal gains as they are given to us by redcal.
        plt.plot(range(self.vis_redcal.nant), np.abs(self.vis_redcal.get_antenna_gains()), "r", linewidth=0.6, label="g_redcal")

        # 3. Sampled gains based on sampled x. 
        plt.plot(range(self.vis_redcal.nant), np.abs(self.vis_sampled.get_antenna_gains()), "b", linewidth=0.6, label="g_sampled")
        
        # Error bars
        assert error_amp.shape[0] == self.vis_redcal.nant
        for i in range(error_amp.shape[0]):
            plt.plot([i, i], [ error_amp[i][0], error_amp[i][1] ], "lightblue")
        plt.legend()
        plt.title("Gain amplitudes")
        plt.xlabel("Antenna")
        plt.ylabel("Amplitude")
        plt.xticks(range(self.nant()), range(self.nant()))

        plt.subplot(2, 1, 2)

        # 1. V true. The gains are 1? Can't remember why
        plt.plot(range(self.vis_redcal.nant), np.angle(self.vis_true.get_antenna_gains()), color="green", linewidth=0.6, label="g_true")

        # 2. The redcal gains as they are given to us by redcal.
        plt.plot(range(self.vis_redcal.nant), np.angle(self.vis_redcal.get_antenna_gains()), "r", linewidth=0.6, label="g_redcal")

        # 3. The sampled gains, actually x is sampled. 
        plt.plot(range(self.vis_redcal.nant), np.angle(self.vis_sampled.get_antenna_gains()), "b", linewidth=0.6, label="g_sampled")
        
        # Error bars
        assert error_phase.shape[0] == self.vis_redcal.nant
        for i in range(error_phase.shape[0]):
            plt.plot([i, i], [ error_phase[i][0], error_phase[i][1] ], "lightblue")
        plt.legend()
        plt.title("Gain phases")
        plt.xlabel("Antenna")
        plt.ylabel("Phase (rad)")
        plt.xticks(range(self.nant()), range(self.nant()))
        plt.tight_layout()

    def assemble_data(self, parameters):
        assert len(parameters) == 2 and ("x" in parameters or "g" in parameters) and "V" in parameters,\
                        "Must be x or g and V."        
        
        data_packet = {}
        if "x" in parameters:            
            data_packet["data"] = self.samples["x"]
            data_packet["x_or_g_len"] = self.samples["x"].shape[1]
        if "g" in parameters:
            data_packet["data"] = self.samples["g"]
            data_packet["x_or_g_len"] = self.samples["g"].shape[1]   
            
        # Now tack V on
        if "V" in parameters:
            data_packet["data"] = np.concatenate((data_packet["data"], self.samples["V"]), axis=1)
        
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

                           
    def x_random_draw(self, v):
        A = reduce_dof(generate_proj(v.g_bar, v.V_model))  # depends on model
        #A = reduce_dof(generate_proj1(v.nvis, v.nant))
        N = np.diag(split_re_im(v.obs_variance))
        d = split_re_im(v.get_reduced_observed())                       # depends on model
        #d = split_re_im(v.get_reduced_observed1()) 
        S = self.S
        
        N_inv = np.linalg.inv(N)
        S_inv = np.linalg.inv(S) 
        A_N_A = np.dot(A.T, np.dot(N_inv, A))
        
        rhs = np.dot(A.T, np.dot(N_inv, d))
        rhs += np.dot(A.T, np.dot(self.sqrtm(N_inv), self.standard_random_draw(A.shape[0])))
        rhs += np.dot(self.sqrtm(S_inv), self.standard_random_draw(A.shape[1]))
        
        bracket_term = S_inv+A_N_A
        
        return np.dot(np.linalg.inv(bracket_term), rhs)
    
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
        
        x = np.dot(np.linalg.inv(bracket_term), rhs)
        
        return np.dot(S_sqrt, x)

    
    def x_random_draw1(self, v):
        A = reduce_dof(generate_proj(v.g_bar, v.V_model))  # depends on model
        #A = reduce_dof(generate_proj1(v.nvis, v.nant))
        N = np.diag(split_re_im(v.obs_variance))
        d = split_re_im(v.get_reduced_observed())                       # depends on model
        #d = split_re_im(v.get_reduced_observed1()) 
        S = self.S
        
        return self.random_draw(np.dot(A.T, np.dot(np.linalg.inv(N), d)), A, S, N)
        
    
    def V_random_draw(self, v):
        A = self.generate_m_proj(v)          
        A = np.dot(A, v.model_projection)
        N = np.diag(split_re_im(v.obs_variance))
        Cv = self.Cv
        d = split_re_im(v.V_obs)
        V_mean = split_re_im(self.V_mean)
                                
        return self.random_draw(np.dot(A.T, np.dot(np.linalg.inv(N), d))+np.dot(np.linalg.inv(Cv), V_mean), A, Cv, N)


    def new_x_distribution(self, v):
        # The model has been updated so get a new distribution.
        # If S is set to None and the model is never changed then
        # the mean of the x distribution will be the GLS solution.

        A = reduce_dof(generate_proj(v.g_bar, v.project_model()))  # depends on model
        #A = reduce_dof(generate_proj1(v.nvis, v.nant))
        N = np.diag(split_re_im(v.obs_variance))
        d = split_re_im(v.get_reduced_observed())                       # depends on model
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
        """
        Form 2-D matrix that can be multiplied by re(model), im(model)
        to give re(V_obs), im(V_obs)
        """

        a = gi.real
        b = gi.imag
        c = gj.real
        d = gj.imag
        e = xi.real
        f = xi.imag
        g = xj.real
        h = xj.imag

        return a*c+b*d + a*c*e+b*d*e-b*c*f+a*d*f + a*c*g+b*d*g+b*c*h-a*d*h, \
                (b*c-a*d) +(a*c*f+b*d*f+b*c*e-a*d*e) + (-a*c*h-b*d*h+b*c*g-a*d*g)
    
    def separate_terms1(self, gi, gj, xi, xj):
        v = gi*np.conj(gj)*(1+xi+np.conj(xj))
        return v.real, v.imag

    def generate_m_proj(self, vis):
        proj = np.zeros((vis.nvis*2, vis.nvis*2))
        k = 0
        for i in range(vis.nant):
            for j in range(i+1, vis.nant):
                #term1, term2 = self.separate_terms(vis.g_bar[i], vis.g_bar[j], vis.x[i], vis.x[j])
                term1, term2 = self.separate_terms1(vis.g_bar[i], vis.g_bar[j], vis.x[i], vis.x[j])
                # Put them in the right place in the bigger matrix
                proj[k*2, k*2] = term1
                proj[k*2, k*2+1] = -term2
                proj[k*2+1, k*2] = term2
                proj[k*2+1, k*2+1] = term1

                k += 1

        # Test - first line equal to second
        #print(split_re_im(vis.get_simulated_visibilities()))
        #print(np.dot(proj, split_re_im(vis.V_model))); exit()

        return proj


    def new_model_distribution(self, v):
        # The x values have been updated so get a new distribution.
        # If the x value has not been changed and C is set to None
        # and N = I then the mean of the distribution will be the 
        # v.V_model

        A = self.generate_m_proj(v)
        A = np.dot(A, v.model_projection)
    
        N = np.diag(split_re_im(v.obs_variance))
        Cv = self.Cv
        d = split_re_im(v.V_obs)
        V_mean = split_re_im(self.V_mean)

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
                data = self.samples[p]
                best_dict[p] = np.mean(data, axis=0)

        elif method == "peak":
             for p in parameters:
                data = self.samples[p]
                best_dict[p] = np.array([ peak(data[:, i]) for i in range(data.shape[1]) ])
                
        elif method == "ml":

            vv = copy.deepcopy(self.vis_redcal)
            assert np.sum(np.abs(vv.x)) == 0, "Redcal object contains x values"
                    
            best = -1e39
            where_best = 0
            for i in range(self.samples["x"].shape[0]):   
                if "x" in parameters:
                    vv.x = unsplit_re_im(restore_x(self.samples["x"][i]))
                else:
                    vv.g_bar = unsplit_re_im(self.samples["g"][i])
                vv.V_model = unsplit_re_im(self.samples["V"][i])
                lh = vv.get_unnormalized_likelihood()
                if lh > best:
                    where_best = i
                    best = lh

            if "x" in parameters:
                best_dict["x"] = self.samples["x"][where_best]

                # Generate gains from best x
                vv.x = unsplit_re_im(restore_x(best_dict["x"]))
                best_dict["g"] = split_re_im(vv.get_antenna_gains())
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
        
        def strip(gains):
            stripped = np.zeros(len(gains.keys()), dtype=type(gains[(0, "Jee")][0, 0]))
            for i, key in enumerate(gains):
                ant = key[0]
                stripped[ant] = gains[key][0, 0]
            return stripped
        
        def fix(cal, gains, gains_dict):
                # Fix degeneracies on gains
            for i in range(self.vis_redcal.nant):
                g = np.empty((1, 1), dtype=type(gains[0]))
                g[0, 0] = gains[i]
                gains_dict[(i, "Jee")] = g

            new_gains = RedCal.remove_degen_gains(gains_dict, 
                                              degen_gains=true_gains, 
                                              mode='complex')
            return strip(new_gains)
        
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
        
           
        true_gains, _ = hc.io.load_cal(self.vis_redcal.file_root+"_g_new.calfits")
        # Same layout as red_gains
        assert len(true_gains.keys()) == self.vis_redcal.nant
        for key in true_gains:
            true_gains[key] = true_gains[key][self.vis_redcal.time:self.vis_redcal.time+1, self.vis_redcal.freq:self.vis_redcal.freq+1]

        # Create calibrator and dict for the work
        RedCal = hc.redcal.RedundantCalibrator(reds)
        gains_dict = {}             # this is just used for temp space
        
        # Fix all the samples
        
        # Now fix the sampled gains and adjust the x values
        self.samples["orig_x"] = np.copy(samples["x"])
        self.samples["orig_g"] = np.copy(samples["g"])
        for i in range(self.samples["orig_g"].shape[0]):
            self.samples["g"][i] = split_re_im(fix(RedCal, unsplit_re_im(self.samples["orig_g"][i]), gains_dict))
            
            # Adjust x
            new_g = unsplit_re_im(self.samples["g"][i])
            orig_g = unsplit_re_im(self.samples["orig_g"][i])
            orig_x = unsplit_re_im(self.samples["orig_x"][i])
            self.samples["x"][i] = split_re_im((orig_g*(1+orig_x))/new_g - 1)
            
  
        # Fix redcal gains
    
        self.vis_redcal.g_bar = fix(RedCal, self.vis_sampled.get_antenna_gains(), gains_dict)
        self.vis_redcal.x.fill(0)            # should be 0 anyway

        # Fix sampled best gains
        
        orig_g = self.vis_sampled.g_bar
        orig_x = self.vis_sampled.x

        self.vis_sampled.g_bar = self.vis_redcal.g_bar      # Update gain
        
        new_g = self.vis_sampled.g_bar
        self.vis_sampled.x = (orig_g*(1+orig_x))/new_g - 1  # Update x
                          
        self.gain_degeneracies_fixed = True
    


if __name__ == "__main__":

    
    sampler = Sampler(seed=99, niter=1000, random_the_long_way=False, best_type="ml")
    sampler.load_nr_sim("/scratch3/users/hgarsden/catall/calibration_points/viscatBC", 
                    freq=0, time=0, remove_redundancy=False, initial_solve_for_x=False)    #sampler.load_sim(3)

    print(sampler.vis_true.get_unnormalized_likelihood(unity_N=True))
    
    S = np.eye(sampler.nant()*2-1)*0.01
    V_mean = sampler.vis_redcal.V_model
    Cv = np.eye(V_mean.size*2)
    sampler.set_S_and_V_prior(S, V_mean, Cv)
    
    sampler.run()
    #sampler.plot_marginals("x", 4)
    print(sampler.vis_redcal.get_unnormalized_likelihood(unity_N=True))
    #sampler.plot_marginals("V", 4)
    exit()

    S = np.eye(sampler.nant()*2-1)*0.01
    V_mean = sampler.vis_redcal.V_model
    Cv = np.eye(V_mean.size*2)
    sampler.set_S_and_V_prior(S, V_mean, Cv)
    sampler.run()
    
    sampler.plot_gains()
