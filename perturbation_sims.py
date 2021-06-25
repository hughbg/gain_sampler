import matplotlib.pyplot as plt
import numpy as np
import sys
from plot import plot_compare
from vis_creator import VisSim, perturb_gains, perturb_model
from gls import gls_solve, reduce_dof, generate_proj
from mcmc import run_mcmc, estimate_variance, loglike
from calcs import split_re_im, unsplit_re_im

def mcmc_sim(perturb_percent, perturb_this, sample_this, num_trials):
    
    def extract_results(samp, prob):
        """
        Extract the best values from the sampling, based on
        likelihoods.
        
        Parameters
        ----------
        samp : array_like
            Shape of samp: ((niter-burn_in)*nwalkers, number of values) 
        prob : array_like
            Shape of prob: ((niter-burn_in)*nwalkers)
        
        Returns
        -------
        array_like
            Array of values.
        """
        where_best = np.argmax(prob) 
        
        vals = np.empty(samp.shape[1])
        for k in range(samp.shape[1]):
            vals[k] = samp[where_best, k]
        
        return vals
    
    def sample_stats(what, samp):
        """
        Print mean/variance of samples
        
        Parameters
        ----------
        samp : array_like
            Shape of samp: ((niter-burn_in)*nwalkers, number of x values) 
        prob : array_like
            Shape of prob: ((niter-burn_in)*nwalkers)
        
        
        """
        for i in range(samp.shape[1]):
            print(what+"_"+str(i)+":", np.mean(samp[:,i]), np.var(samp[:,i]))
            

    print("mcmc sim, perturb", perturb_this, "by", str(perturb_percent)+"%")
    NANT = 4
    NBASELINE = 6

    if perturb_this not in [ "gain", "model" ]:
        raise ValueError("\"perturb_this\" has unknown value "+perturb_this)

    if sample_this not in [ "x", "model" ]:
        raise ValueError("\"sample_this\" has unknown value "+sample_this)

    for i in range(num_trials):

        print(i, end=" "); sys.stdout.flush()

        vis_values = VisSim(NANT, level="approx", random_seed=99)
        
        if sample_this == "x":
            proj, _ = reduce_dof(generate_proj(vis_values.g_bar, vis_values.V_model), True)
            data_vector = split_re_im(vis_values.get_reduced_observed())
        else: 
            proj = np.eye(vis_values.V_model.size*2)
            data_vector = split_re_im(vis_values.get_calibrated_visibilities())

        if i == 0: 
            val_in = np.empty((num_trials, vis_values.V_obs.size), np.complex64)
            val_out = np.empty((num_trials,vis_values.V_obs.size), np.complex64)

        # This is what we want to compare against for assessing quality of mcmc results
        if sample_this == "x":
            val_in[i] = vis_values.get_baseline_gains()     # save
        else:
            val_in[i] = vis_values.V_obs
 
        if perturb_this == "model":
            p_vis_values = perturb_model(vis_values, perturb_percent)
        else:
            p_vis_values = perturb_gains(vis_values, perturb_percent)
        
        # Run MCMC sampling
        samp, prob = run_mcmc(proj, data_vector, np.repeat(p_vis_values.obs_variance, 2), 10000)
        print()
        
        # Get best values

        if sample_this == "x":
            sample_stats("x", samp)
            p_vis_values.x = unsplit_re_im(np.append(extract_results(samp, prob), 0))
            val_out[i] = p_vis_values.get_baseline_gains()
            initial = "g"
        else:
            sample_stats("model", samp)
            p_vis_values.V_model = unsplit_re_im(extract_results(samp, prob))
            val_out[i] = p_vis_values.get_simulated_visibilities()
            initial = "m"

        print(val_in[i])
        print(val_out[i])
        plot_compare(val_in[:i+1], val_out[:i+1],
                    initial+"_approx_hist_mcmc_"+str(perturb_percent)+".png", initial+"_in", initial+"_out")

    print()



def gls_sim(perturb_percent, perturb_this, num_trials):

    NANT = 4
    NBASELINE = 6

    print("gls sim, perturb", perturb_this, "by", str(perturb_percent)+"%")
    
    val_in = np.empty((num_trials, NBASELINE), dtype=np.complex64)
    val_out = np.empty((num_trials, NBASELINE), dtype=np.complex64)

    for i in range(num_trials):

        print(i, end=" "); sys.stdout.flush()

        vis_values = VisSim(NANT, level="approx")

        var_mc = np.eye(vis_values.V_obs.size)
  
        val_in[i] = vis_values.get_baseline_gains()

        # Alter something so the gains/x values will be wrong, and x has to be found again
        if perturb_this == "gain": p_vis_values = perturb_gains(vis_values, perturb_percent)
        elif perturb_this == "vis": p_vis_values = perturb_vis(vis_values, perturb_percent)
        else:
            raise ValueError("What to perturb is not properly specified")

        # Find new x values
        p_vis_values.x = gls_solve(p_vis_values)
            
        val_out[i] = p_vis_values.get_baseline_gains()          
            
        print("val_in", val_in[i], "\nval_out", val_out[i], "\n Diff", np.mean(np.abs((val_in[i]-val_out[i])/val_in[i])*100))
        plot_compare(val_in[:i+1], val_out[:i+1], "g_hist_gls_"+str(perturb_percent)+".png", "g_in", "g_out")
 
    print()

if __name__ == "__main__":
    
    # The perturbation sims are not suitable for running in a notebook.
    # Run on a cluster.
    
    from multiprocessing import Process

    mcmc_sim(0, "gain", "model", 1); exit()
    """
    # GLS
    processes = []
    for perturb_percent in [0, 1, 5, 10, 15, 20, 30, 40, 50, 75, 90 ]:
        p = Process(target=gls_sim, args=(perturb_percent, "gain", 1000))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    """

    # MCMC using exact gain calculation
    processes = []
    for perturb_percent in [ 0, 1, 5, 10, 20, 30, 40, 50, 75, 90 ]:
        p = Process(target=mcmc_sim, args=(perturb_percent, "gain", 1000, "exact"))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # MCMC using approximate gain calculation
    processes = []
    for perturb_percent in [ 0, 1, 5, 10, 20, 30, 40, 50, 75, 90 ]:
        p = Process(target=mcmc_sim, args=(perturb_percent, "gain", 1000, "approx"))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
