import matplotlib.pyplot as plt
from vis_creator import VisSim, VisCal, VisTrue
from gls import gls_solve, generate_proj, generate_proj1, reduce_dof, restore_x
from calcs import split_re_im, unsplit_re_im
import copy
import numpy as np

class Sampler:
    
    def __init__(self, niter=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.niter = niter
    
    def load_nr_sim(self, path, time=0, freq=0, initial_solve_for_x=False):
        print("Loading NR sim from", path)
        self.vis_sim = VisCal(path, time=time, freq=freq)
        self.vis_sim_true = VisTrue(path, time=time, freq=freq)

        if initial_solve_for_x:
            self.vis_sim.x = self.vis_sim_true.initial_vals.x = gls_solve(self.vis_sim)
            
    def load_sim(self, nant, initial_solve_for_x=False):
        self.vis_sim = VisSim(nant)
        self.vis_sim_true = self.vis_sim
        if initial_solve_for_x:
            self.vis_sim.x = self.vis_sim_true.initial_vals.x = gls_solve(self.vis_sim)
            
    def set_S_and_V(self, S, V_mean, Cv):
        self.S = S
        self.V_mean = V_mean
        self.Cv = Cv
        
    def nant(self):
        return self.vis_sim.nant
    
    def nvis(self):
        return self.vis_sim.nvis
        
            
    def run(self):
        print("Running sampling")
        all_x = np.zeros((self.niter, self.vis_sim.nant*2-1))       # -1 because there'll be a missing imaginary value
        all_model = np.zeros((self.niter, self.vis_sim.nvis*2))

        v_x_sampling = copy.deepcopy(self.vis_sim)      
        v_model_sampling = copy.deepcopy(self.vis_sim) 

        new_x_sample = v_model_sampling.x         # Initialize

        # Take num samples
        for i in range(self.niter):
            # Use the sampled x to change the model sampling distribution, and take a sample
            v_model_sampling.x = new_x_sample
            v_dist_mean, v_dist_covariance = self.new_model_distribution(v_model_sampling)
            all_model[i] = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)
            new_model_sample = unsplit_re_im(all_model[i])
            #new_model_sample[:] = np.mean(new_model_sample)      # Redundant only

            # Use the sampled model to change the x sampling distribution, and take a sample
            v_x_sampling.V_model = new_model_sample
            x_dist_mean, x_dist_covariance = self.new_x_distribution(v_x_sampling)
            all_x[i] = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  
            new_x_sample = unsplit_re_im(restore_x(all_x[i]))
            
        self.all_x = all_x
        self.all_model = all_model
    
    
    def plot_marginals(self):
        def plot_hist(a, fname, label, sigma_prior, other_vals):
            hist, bin_edges = np.histogram(a, bins=len(a)//50)
            bins = (bin_edges[1:]+bin_edges[:-1])/2

            sigma = np.std(a-np.mean(a))

            plt.clf()

            plt.plot(bins, hist)
            for key in other_vals:
                plt.axvline(other_vals[key][0], color=other_vals[key][1], label=key)
            plt.title(label+" sigma: "+str(round(sigma,2))+" sigma_prior: "+str(round(sigma_prior,2)))
            plt.legend()
            plt.savefig(fname+".png")

       
        orig_x = split_re_im(self.vis_sim.x)
        true_x = split_re_im((self.vis_sim_true.g_bar-self.vis_sim.g_bar)/self.vis_sim.g_bar)
        S_sigmas = np.sqrt(np.diag(self.S))            
        for i in range(self.vis_sim.nant*2-1):
            if i%2 == 0: part = "re"
            else: part = "im"
            other_vals = {
                "Orig": ( orig_x[i], "r" ),
                "True": ( true_x[i], "g" )
            }
            print("<img src="+part+"_x_"+str(i//2)+".png width=400>")
            plot_hist(self.all_x[:, i], part+"_x_"+str(i//2), part+"(x_"+str(i//2)+")", S_sigmas[i], other_vals)
        
        true_model = split_re_im(self.vis_sim_true.V_model)
        redcal_model = split_re_im(self.vis_sim.V_model)
        V_mean = split_re_im(self.V_mean)
        Cv_sigmas = np.sqrt(np.diag(self.Cv))
        for i in range(self.vis_sim.nvis*2):
            if i%2 == 0: part = "re"
            else: part = "im"
            other_vals = {
                "Redcal" : ( redcal_model[i], "r" ),
                "True": ( true_model[i], "g" )
            }
            print("<img src="+part+"_V_"+str(i//2)+".png width=400>")
            plot_hist(self.all_model[:, i], part+"_V_"+str(i//2), part+"(V_"+str(i//2)+")", Cv_sigmas[i], other_vals)
        
    def new_x_distribution(self, v):
        # The model has been updated so get a new distribution.
        # If S is set to None and the model is never changed then
        # the mean of the x distribution will be the GLS solution.

        A = reduce_dof(generate_proj(v.g_bar, v.V_model))  # depends on model
        A = reduce_dof(generate_proj1(v.nvis, v.nant))
        N = np.diag(split_re_im(v.obs_variance))
        d = split_re_im(v.get_reduced_observed())                       # depends on model
        d = split_re_im(v.get_reduced_observed1()) 
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
                -(b*c-a*d) -(a*c*f+b*d*f+b*c*e-a*d*e) - (-a*c*h-b*d*h+b*c*g-a*d*g)

    def generate_m_proj(self, vis):
        proj = np.zeros((vis.nvis*2, vis.nvis*2))
        k = 0
        for i in range(vis.nant):
            for j in range(i+1, vis.nant):
                term1, term2 = self.separate_terms(vis.g_bar[i], vis.g_bar[j], vis.x[i], vis.x[j])
                # Put them in the right place in the bigger matrix
                proj[k*2, k*2] = term1
                proj[k*2, k*2+1] = term2
                proj[k*2+1, k*2] = -term2
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

        N = np.diag(split_re_im(v.obs_variance))
        Cv = self.Cv
        d = split_re_im(v.V_obs)
        V_mean = self.V_mean

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

        
if __name__ == "__main__":
    sampler = Sampler(seed=99, niter=40000)
    sampler.load_nr_sim("/scratch2/users/hgarsden/catall/calibration_points/viscatBC_stretch0.02", initial_solve_for_x=False)
    #sampler.load_sim(4)
    S = np.eye(sampler.nant()*2-1)*0.01
    V_mean = split_re_im(sampler.vis_sim.V_model)
    Cv = np.eye(V_mean.size)
    sampler.set_S_and_V(S, V_mean, Cv)
    sampler.run()
    sampler.plot_marginals()
