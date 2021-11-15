import matplotlib.pyplot as plt
from vis_creator import VisSim, VisCal, VisTrue
from gls import gls_solve, generate_proj, generate_proj1, reduce_dof, restore_x
from calcs import split_re_im, unsplit_re_im
import copy
import numpy as np
import scipy.linalg

class Sampler:
    
    def __init__(self, niter=1000, seed=None, random_the_long_way=False):
        if seed is not None:
            np.random.seed(seed)
        self.niter = niter
        self.random_the_long_way = random_the_long_way
    
    def load_nr_sim(self, path, time=0, freq=0, remove_redundancy=False, initial_solve_for_x=False):
        print("Loading NR sim from", path)
        self.vis_sim = VisCal(path, time=time, freq=freq, remove_redundancy=remove_redundancy)
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
        all_model = np.zeros((self.niter, self.V_mean.size*2))

        v_x_sampling = copy.deepcopy(self.vis_sim)      
        v_model_sampling = copy.deepcopy(self.vis_sim) 

        new_x = v_model_sampling.x         # Initialize
        
        # Take num samples
        for i in range(self.niter):
            # Use the sampled x to change the model sampling distribution, and take a sample
            v_model_sampling.x = new_x
            if self.random_the_long_way:
                all_model[i] = self.V_random_draw(v_model_sampling)
            else:
                v_dist_mean, v_dist_covariance = self.new_model_distribution(v_model_sampling)
                all_model[i] = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)
            
            new_model = unsplit_re_im(np.dot(v_model_sampling.model_projection, all_model[i]))
            #new_model_sample[:] = np.mean(new_model_sample)      # Redundant only

            # Use the sampled model to change the x sampling distribution, and take a sample
            v_x_sampling.V_model = new_model
            if self.random_the_long_way:
                all_x[i] = self.x_random_draw1(v_x_sampling)
            else:
                x_dist_mean, x_dist_covariance = self.new_x_distribution(v_x_sampling)
                all_x[i] = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  
            
            new_x = unsplit_re_im(restore_x(all_x[i]))
            
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
        for i in range(self.all_x.shape[1]):
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
        V_mean = np.dot(self.vis_sim.model_projection, split_re_im(self.V_mean))
        Cv_sigmas = np.sqrt(np.diag(self.Cv))
        for i in range(self.all_model.shape[1]):
            if i%2 == 0: part = "re"
            else: part = "im"
            other_vals = {
                "Redcal" : ( redcal_model[i], "r" ),
                "True": ( true_model[i], "g" )
            }
            print("<img src="+part+"_V_"+str(i//2)+".png width=400>")
            plot_hist(self.all_model[:, i], part+"_V_"+str(i//2), part+"(V_"+str(i//2)+")", Cv_sigmas[i], other_vals)
            
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
        all_x = np.zeros((self.niter, self.vis_sim.nant*2-1))       # -1 because there'll be a missing imaginary value
        all_model = np.zeros((self.niter, self.V_mean.size*2))

        for i in range(self.niter):
            if self.random_the_long_way:
                all_x[i] = self.x_random_draw(self.vis_sim)
                all_model[i] = self.V_random_draw(self.vis_sim)
            else:
                x_dist_mean, x_dist_covariance = self.new_x_distribution(self.vis_sim)
                all_x[i] = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  

                v_dist_mean, v_dist_covariance = self.new_model_distribution(self.vis_sim)
                all_model[i] = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)    
                
        self.all_x = all_x
        self.all_model = all_model

                           
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

        A = reduce_dof(generate_proj(v.g_bar, v.V_model))  # depends on model
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
            best_x = np.array([ np.mean(self.all_x[:, i]) for i in range(self.all_x.shape[1]) ])
            best_model = np.array([ np.mean(self.all_model[:, i]) for i in range(self.all_model.shape[1]) ])    

        elif method == "hist":
            best_x = np.array([ peak(self.all_x[:, i]) for i in range(self.all_x.shape[1]) ])
            best_model = np.array([ peak(self.all_model[:, i]) for i in range(self.all_model.shape[1]) ])   

        elif method == "ml":
            where_best, _ = self.fit_stat()
            best_x = self.all_x[where_best]
            best_model = self.all_model[where_best]

        else:
            raise ValueError("Invalid method")

        return best_x, best_model
    
    def fit_stat(self):
        
        vv = copy.deepcopy(self.vis_sim)
        vv.g_bar = vv.initial_vals.g_bar
        vv.V_model = vv.initial_vals.g_bar

        best = 1e39
        where_best = 0
        for i in range(self.all_x.shape[0]):
            vv.x = unsplit_re_im(restore_x(self.all_x[i]))
            vv.V_model = unsplit_re_im(np.dot(vv.model_projection, self.all_model[i]))
            chi2 = vv.get_quality()
            if chi2 < best:
                where_best = i
                best = chi2

        return where_best, best


        
if __name__ == "__main__":
    sampler = Sampler(seed=99, niter=10000, random_the_long_way=False)
    sampler.load_nr_sim("/scratch2/users/hgarsden/catall/calibration_points/viscatBC_stretch0.02", remove_redundancy=True, initial_solve_for_x=False)

    #sampler.load_sim(4)
    S = np.eye(sampler.nant()*2-1)*0.01
    V_mean = sampler.vis_sim.group_models()
    Cv = np.eye(V_mean.size*2)
    sampler.set_S_and_V(S, V_mean, Cv)
    sampler.run()
    sampler.plot_marginals()
    _, stat = sampler.fit_stat()
    print(stat, sampler.vis_sim.get_quality())

