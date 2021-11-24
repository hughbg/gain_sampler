import matplotlib.pyplot as plt
from vis_creator import VisSim, VisCal, VisTrue
from gls import gls_solve, generate_proj, generate_proj1, reduce_dof, restore_x
from calcs import split_re_im, unsplit_re_im
import corner
import copy
import numpy as np
import scipy.linalg



class Sampler:
    
    def __init__(self, niter=1000, burn_in=10, seed=None, random_the_long_way=False):
        if seed is not None:
            np.random.seed(seed)
        self.niter = niter
        self.burn_in = burn_in
        self.random_the_long_way = random_the_long_way
    
    def load_nr_sim(self, path, time=0, freq=0, remove_redundancy=False, initial_solve_for_x=False):
        print("Loading NR sim from", path)
        self.vis_redcal = VisCal(path, time=time, freq=freq, remove_redundancy=remove_redundancy)
        self.vis_true = VisTrue(path, time=time, freq=freq)

        if initial_solve_for_x:
            self.vis_redcal.x = self.vis_redcal.initial_vals.x = gls_solve(self.vis_redcal)
            
    def load_sim(self, nant, initial_solve_for_x=False, **kwargs):
        self.vis_redcal = VisSim(nant, **kwargs)
        self.vis_true = self.vis_redcal
        if initial_solve_for_x:
            self.vis_redcal.x = self.vis_redcal.initial_vals.x = gls_solve(self.vis_redcal)
            
    def set_S_and_V_prior(self, S, V_mean, Cv):
        self.S = S
        self.V_mean = V_mean
        self.Cv = Cv
        
    def nant(self):
        return self.vis_redcal.nant
    
    def nvis(self):
        return self.vis_redcal.nvis
                   
    def run(self):
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
                        
        self.sampled_x = sampled_x[(self.niter*self.burn_in)//100:]
        self.sampled_V = sampled_V[(self.niter*self.burn_in)//100:]
                
        self.vis_sampled = copy.deepcopy(self.vis_redcal)    
        self.best_x, self.best_model = self.bests(method="ml")
        self.vis_sampled.x = unsplit_re_im(restore_x(self.best_x))
        self.vis_sampled.V_model = unsplit_re_im(self.best_model)
        
        self.x_means = np.mean(sampled_x, axis=0)
        self.model_means = np.mean(sampled_V, axis=0)

    
    
    def plot_marginals(self, what, cols):
        def plot_hist(a, fname, label, sigma_prior, other_vals, index):
            hist, bin_edges = np.histogram(a, bins=len(a)//50)
            bins = (bin_edges[1:]+bin_edges[:-1])/2

            sigma = np.std(a-np.mean(a))

            plt.subplot(rows, cols, index)
            plt.plot(bins, hist)
            for key in other_vals:
                plt.axvline(other_vals[key][0], color=other_vals[key][1], label=key)
            plt.title(label+" sigma: "+str(round(sigma,2))+" sigma_prior: "+str(round(sigma_prior,2)))
            plt.legend()

        if what == "x":
            num_plots = self.sampled_x.shape[1]
            if num_plots%cols == 0: rows = num_plots//cols
            else: rows = num_plots//cols+1
            
            orig_x = split_re_im(self.vis_redcal.x)
            true_x = split_re_im((self.vis_true.g_bar-self.vis_redcal.g_bar)/self.vis_redcal.g_bar)
            S_sigmas = np.sqrt(np.diag(self.S))            
            for i in range(self.sampled_x.shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {
                    "Orig": ( orig_x[i], "r" ),
                    "True": ( true_x[i], "k" )
                }

                plot_hist(self.sampled_x[:, i], part+"_x_"+str(i//2), part+"(x_"+str(i//2)+")", S_sigmas[i], other_vals, i+1)
                
        elif what == "V":
            true_model = split_re_im(self.vis_true.V_model)
            redcal_model = split_re_im(self.vis_redcal.V_model)
            V_mean = np.dot(self.vis_redcal.model_projection, split_re_im(self.V_mean))
            Cv_sigmas = np.sqrt(np.diag(self.Cv))
            for i in range(self.sampled_V.shape[1]):
                if i%2 == 0: part = "re"
                else: part = "im"
                other_vals = {
                    "Redcal" : ( redcal_model[i], "r" ),
                    "True": ( true_model[i], "k" )
                }
                plot_hist(self.sampled_V[:, i], part+"_V_"+str(i//2), part+"(V_"+str(i//2)+")", Cv_sigmas[i], other_vals)
        else:
            raise ValueError("Invalid spec for plot_marginals")
            
        plt.tight_layout()
        
    def plot_corner(self):
        part = lambda i : "re" if i%2==0 else "im"
        data = np.concatenate((self.sampled_x, self.sampled_V), axis=1)
        
        # Plot x dist
        
        labels = [ r"$"+part(i)+"(x_"+str(i)+")$" for i in range(self.sampled_x.shape[1]) ]+  \
                    [ r"$"+part(i)+"(V_"+str(i)+")$" for i in range(self.sampled_V.shape[1]) ]
        figure = corner.corner(data, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)
        
        axes = np.array(figure.axes).reshape((data.shape[1], data.shape[1]))

        # Loop over the diagonal
        orig_x = reduce_dof(split_re_im(self.vis_redcal.x))
        orig_model = split_re_im(self.vis_redcal.V_model)
        orig = np.concatenate((orig_x, orig_model))

        for i in range(orig.size):
            #print(i, orig[i])
            ax = axes[i, i]
            ax.axvline(orig[i], color="r", linewidth=0.5)

        plt.tight_layout()

        #return figure
        
    def plot_trace(self, what):
        if what == "x": 
            data = self.sampled_x
        elif what == "V":
            data = self.sampled_V
        else:
            raise ValueError("plot_trace has invalid specification")
            
        for i in range(data.shape[1]):
            plt.plot(data[:, i])
            
        plt.xlabel("Sample iteration")
        plt.ylabel(what)
        
            
    def print_covcorr(self, threshold=0.0, list_them=False):
        m = self.corrcoef()
        part = lambda i : "re" if i%2==0 else "im"
        labels = [ part(i)+"(x_"+str(i)+")" for i in range(self.sampled_x.shape[1]) ]+  \
                    [ part(i)+"(V_"+str(i)+")" for i in range(self.sampled_V.shape[1]) ]
        
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
            
    def plot_covcorr(self, threshold=0.0):
        m = self.corrcoef()
        np.fill_diagonal(m, 0)
        m[np.triu_indices(m.shape[0])] = 0
        m[np.logical_and(m>-threshold,m<threshold)] = 0

        
        plt.subplot(2, 2, 1)
        plt.imshow(m, cmap="RdBu", aspect="auto")
        nx = self.sampled_x.shape[1]
        nv = self.sampled_V.shape[1]
        x=plt.xticks([nx//2, nx+nv//2], ["X", "V"], fontsize=15)
        x=plt.yticks([nx//2, nx+nv//2], ["X", "V"], fontsize=15)
        plt.plot([-0.5, nx-0.5], [nx-0.5, nx-0.5], "k", linewidth=0.6)
        plt.plot([nx-0.5, nx-0.5], [-0.5, nx-0.5],  "k", linewidth=0.6)
        plt.colorbar()
        plt.title("Correlation matrix")
        
        
        plt.subplot(2, 2, 2)
        plt.plot(range(self.sampled_x.shape[1]), self.x_means, label="Means", linewidth=0.6)
        plt.plot(range(self.sampled_x.shape[1]), self.best_x, label="MLs", linewidth=0.6)
        x=plt.xticks(range(0, self.sampled_x.shape[1], 2), range(0, self.sampled_x.shape[1], 2))
        plt.xlabel("Order")
        plt.legend()
        plt.title("Best values x split real/imag in order of correlation matrix")
        
        plt.subplot(2, 2, 3)
        plt.plot(range(self.sampled_V.shape[1]), self.model_means, label="Means", linewidth=0.6)
        plt.plot(range(self.sampled_V.shape[1]), self.best_model, label="MLs", linewidth=0.6)
        plt.xlabel("Order")
        plt.legend()
        plt.title("Best values V split real/imag in order of correlation matrix")

        
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
        
        def get_g_error_bars():
            SIG = 3
            BOTTOM = 0
            TOP = 1
            
            sampled_gains = np.zeros((self.sampled_x.shape[0], self.nant))
            for i in range(self.sampled_x.shape[0]):
                x_vec = unsplit_re_im(restore_x(self.sampled_x[i]))
                sampled_gains[i] = self.vis_redcal.g_bar*(1+new_x)
                                 
            
            g_limits_amp = np.zeros((self.sampled_x.shape[1], 2))    # 2 because mean/ml
            g_limits_phase = np.zeros((self.sampled_x.shape[1], 2))
            
            # Get the error bars. For each g, get the range based on SIG sigma.
            for i in range(self.sampled_x.shape[1]):       # Loop over antennas
                m = np.mean(np.abs(sampled_gains[:, i]))
                s = np.std(np.abs(sampled_gains[:, i]))
                g_limits_amp[i, BOTTOM] = m-SIG*s
                g_limits_amp[i, TOP] = m+SIG*s
                
                m = np.mean(np.angle(sampled_gains[:, i]))
                s = np.std(np.angle(sampled_gains[:, i]))
                g_limits_phase[i, BOTTOM] = m-SIG*s
                g_limits_phase[i, TOP] = m+SIG*s
                
                
            return g_limits_amp, g_limits_phase
        
        
        #g_bar = self.v_sampled.g_bar
        #print(get_x_error_bars())
        #exit()
        
        error_amp, error_phase = get_g_error_bars()
        
        plt.subplot(2, 1, 1)

        # 1. V true. The gains are 1? Can't remember why
        plt.plot(range(self.vis_redcal.nant), np.abs(self.vis_true.get_antenna_gains()), "k", label="g_true")

        # 2. The redcal gains as they are given to us by redcal.
        plt.plot(range(self.vis_redcal.nant), np.abs(self.vis_redcal.get_antenna_gains()), "r", label="g_redcal")

        # 3. Sampled gains based on sampled x. 
        plt.plot(range(self.vis_redcal.nant), np.abs(self.vis_sampled.get_antenna_gains()), "b", label="g_sampled")
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
        plt.plot(range(self.vis_redcal.nant), np.angle(self.vis_true.get_antenna_gains()), "k", label="g_true")

        # 2. The redcal gains as they are given to us by redcal.
        plt.plot(range(self.vis_redcal.nant), np.angle(self.vis_redcal.get_antenna_gains()), "r", label="g_redcal")

        # 3. The sampled gains, actually x is sampled. 
        plt.plot(range(self.vis_redcal.nant), np.angle(self.vis_sampled.get_antenna_gains()), "b", label="g_sampled")
        assert error_phase.shape[0] == self.vis_redcal.nant
        for i in range(error_phase.shape[0]):
            plt.plot([i, i], [ error_phase[i][0], error_phase[i][1] ], "lightblue")
        plt.legend()
        plt.title("Gain phases")
        plt.xlabel("Antenna")
        plt.ylabel("Phase (rad)")
        plt.xticks(range(self.nant()), range(self.nant()))
        plt.tight_layout()


        
    def cov(self):
        data = np.concatenate((self.sampled_x, self.sampled_V), axis=1)

        return np.cov(data, rowvar=False)
    
    def corrcoef(self):
        data = np.concatenate((self.sampled_x, self.sampled_V), axis=1)

        return np.corrcoef(data, rowvar=False) 
            
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
    
    def bests(self, method="mean"):
        def peak(a):
            # Get the peak 
            hist, bin_edges = np.histogram(a, bins=len(a)//10)
            bins = (bin_edges[1:]+bin_edges[:-1])/2
            return bins[np.argmax(hist)]


        if method == "mean":
            best_x = np.array([ np.mean(self.sampled_x[:, i]) for i in range(self.sampled_x.shape[1]) ])
            best_model = np.array([ np.mean(self.sampled_V[:, i]) for i in range(self.sampled_V.shape[1]) ])    

        elif method == "hist":
            best_x = np.array([ peak(self.sampled_x[:, i]) for i in range(self.sampled_x.shape[1]) ])
            best_model = np.array([ peak(self.sampled_V[:, i]) for i in range(self.sampled_V.shape[1]) ])   

        elif method == "ml":
            where_best = self.fit_stat()
            best_x = self.sampled_x[where_best]
            best_model = self.sampled_V[where_best]

        else:
            raise ValueError("Invalid method")

        return best_x, best_model
    
    def fit_stat(self):
        
        vv = copy.deepcopy(self.vis_redcal)
        
        best = 1e39
        where_best = 0
        for i in range(self.sampled_x.shape[0]):    
            vv.x = unsplit_re_im(restore_x(self.sampled_x[i]))
            vv.V_model = unsplit_re_im(self.sampled_V[i])
            lh = vv.get_likelihood()
            if lh < best:
                where_best = i
                best = lh
                

        return where_best
    
    


if __name__ == "__main__":

    
    sampler = Sampler(seed=99, niter=1000, random_the_long_way=False)
    sampler.load_nr_sim("/scratch3/users/hgarsden/catall/calibration_points/viscatBC", 
                        remove_redundancy=False, initial_solve_for_x=False)
    S = np.eye(sampler.nant()*2-1)*0.01
    V_mean = sampler.vis_redcal.V_model
    Cv = np.eye(V_mean.size*2)
    sampler.set_S_and_V_prior(S, V_mean, Cv)
    
    sampler.run()
    sampler.plot_covcorr()
    exit()

    S = np.eye(sampler.nant()*2-1)*0.01
    V_mean = sampler.vis_redcal.V_model
    Cv = np.eye(V_mean.size*2)
    sampler.set_S_and_V_prior(S, V_mean, Cv)
    sampler.run()
    
    sampler.plot_gains()
