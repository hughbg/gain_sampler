import matplotlib.pyplot as plt
import numpy as np
import sys
from plot import plot_compare
from vis_creator import VisSim, perturb_gains, perturb_vis
from gls import gls_solve
from mcmc import run_mcmc, estimate_variance, loglike

def mcmc_sim(perturb_percent, perturb_this, num_trials, level):
    
    def extract_x_results(samp, prob):
        """
        Extract the best "x" values from the sampling, based on
        likelihoods.
        
        Parameters
        ----------
        samp : array_like
            Shape of samp: ((niter-burn_in)*nwalkers, number of x values) 
        prob : array_like
            Shape of prob: ((niter-burn_in)*nwalkers)
        
        Returns
        -------
        array_like
            Array of x values.
        """
        where_best = np.argmax(prob[int(prob.size*0.6):]) # Ignore first 60%
        
        vals = np.empty(samp.shape[1])
        for k in range(samp.shape[1]):
            vals[k] = samp[where_best, k]
        
        return vals

    print("mcmc sim, perturb", perturb_this, "by", str(perturb_percent)+"%")
    NANT = 4
    NBASELINE = 6

    if perturb_this not in [ "gain", "vis" ]:
        raise ValueError("\"perturb_this\" has unknown value")

    if level not in [ "exact", "approx" ]:
        raise ValueError("Invalid level in mcmc")


    # Estimate the variance that will be generated
    variance = estimate_variance(perturb_percent, perturb_this, level)

    for i in range(num_trials):

        print(i, end=" "); sys.stdout.flush()

        vis_values = VisSim(NANT, level=level)
        
        if i == 0: 
            val_in = np.empty((num_trials, vis_values.V_obs.size), np.complex64)
            val_out = np.empty((num_trials,vis_values.V_obs.size), np.complex64)
            var_mc = np.full(vis_values.V_obs.size, variance)
                
        val_in[i] = vis_values.get_baseline_gains()     # save
 
        if perturb_this == "vis":
            p_vis_values = perturb_vis(vis_values, perturb_percent)
        else:
            p_vis_values = perturb_gains(vis_values, perturb_percent)
        
        # Run MCMC sampling
        samp, prob = run_mcmc(p_vis_values, var_mc, 2000)

        # Get best values
        p_vis_values.x = extract_x_results(samp, prob)

        val_out[i] = p_vis_values.get_baseline_gains()

        if level == "exact":
            plot_compare(val_in[:i+1], val_out[:i+1],
                    "g_hist_mcmc_"+str(perturb_percent)+".png", "g_in", "g_out")
        else:
            plot_compare(val_in[:i+1], val_out[:i+1],
                    "g_approx_hist_mcmc_"+str(perturb_percent)+".png", "g_in", "g_out")

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
        gls_solve(p_vis_values)
            
        val_out[i] = p_vis_values.get_baseline_gains()          
            
        #print("val_in", val_in[i], "\nval_out", val_out[i], "\n Diff", np.mean(np.abs((val_in[i]-val_out[i])/val_in[i])*100))
        plot_compare(val_in[:i+1], val_out[:i+1], "g_hist_gls_"+str(perturb_percent)+".png", "g_in", "g_out")
 
    print()

if __name__ == "__main__":
    
    # The perturbation sims are not suitable for running in a notebook.
    # Run on a cluster.
    
    from multiprocessing import Process

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
