import numpy as np
import emcee
import sys
import sympy as sp
from plot import plot_x_hist
from vis_creator import VisSim, perturb_gains, perturb_vis


def approximate_variance(perturb_percent, perturb_this, level):
    num_samples = 100
    num_vis = VisSim(num_samples).proj.shape[0]
    all_dy_values = np.zeros(num_vis*num_samples)
    for i in range(num_samples):
        vis_values = VisSim(num_samples, level=level)
        if perturb_this == "vis":
            vis_values = perturb_vis(vis_values, perturb_percent)
        else:
            vis_values = perturb_gains(vis_values, perturb_percent)
        all_dy_values[i*num_vis: i*num_vis+num_vis] = vis_values.calculate_dy()
    
    return np.mean(all_dy_values**2)

def approximate_variance1(samples, vis_values, logfunc):
    var_vec = [ 1e3 for i in range(vis_values.proj.shape[0]) ]
    all_dys = []
    for i in range(samples.shape[0]):
        dys = logfunc(samples[i], vis_values, var_vec, True)
        all_dys += dys
    return np.mean(np.array(all_dys)**2)

    
def exact_loglike(x, vis_values, var_vec, return_dy=False):
    V_model = vis_values.V_model    # Model vis also need to perturb the model
                                 # But need to reduce degrees of freedom - CorrCal
    V_observed = vis_values.V
    g_bar = vis_values.g_bar
    
    dchi2 = 0
    k = -1
    dys = []
    for i in range(g_bar.size):
        for j in range(i+1, g_bar.size):
            k += 1
            #print(i,j,k)
            # Includes quadratic terms in the gs.

            # Calculate the difference in the observed visibility using the proposed
            # gain offsets g_vec.
            dy = g_bar[i] * g_bar[j]*(1 + x[i])*(1 + x[j]) * V_model[k] - V_observed[k]
            dys.append(dy)        
            # Calculate the log probablity of this dy, assuming dy goes by a Gaussian distribution
            # with 0 mean and a variance of var_vec[k]
            dchi2 += -0.5 * dy**2 * (1./var_vec[k])   #- np.log(var_vec[k]) - 0.5*np.log(2.*np.pi)
    if return_dy: return dys
    return dchi2

def approx_loglike(x, vis_values, var_vec, return_dy=False):
    V_model = vis_values.V_model    # Model vis also need to perturb the model
                                 # But need to reduce degrees of freedom - CorrCal
    V_observed = vis_values.V
    g_bar = vis_values.g_bar

    dchi2 = 0
    k = -1
    dys = []
    for i in range(g_bar.size):
        for j in range(i+1, g_bar.size):
            k += 1
            #print(i,j,k)

            # Calculate the difference in the observed visibility using the proposed
            # gain offsets g_vec.
            dy = g_bar[i] * g_bar[j]*(1 + x[i] + x[j]) * V_model[k] - V_observed[k]
            dys.append(dy)
            # Calculate the log probablity of this dy, assuming dy goes by a Gaussian distribution
            # with 0 mean and a variance of var_vec[k]
            dchi2 += -0.5 * dy**2 * (1./var_vec[k])   #- np.log(var_vec[k]) - 0.5*np.log(2.*np.pi)
    if return_dy: return dys
    return dchi2



def run_mcmc(vis_values, variances, loglike=exact_loglike, iters=1000):

    # Set up emcee sampler
    ndim = vis_values.g_bar.size
    nwalkers = ndim * 3
    p0 = 0.01 * np.random.rand(nwalkers, ndim)

    # Set-up sampler object
    sampler = emcee.EnsembleSampler(nwalkers, ndim, exact_loglike, args=[vis_values, variances])

    # Run sampler
    for sample in sampler.sample(p0, iterations=iters, progress=False):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

    # Return samples
    burn_in = int(iters*0.6)
    samples = sampler.get_chain(flat=True, discard=burn_in)
    prob = sampler.get_log_prob(flat=True, discard=burn_in)

    """
    samp = sampler.get_chain()
    
    plt.clf()
    for j in range(samp.shape[1]):
        for k in range(samp.shape[2]):
            plt.plot(samp[:, j, k])
    plt.xlabel("iter", fontsize=15)
    plt.ylabel("x", fontsize=15)
    plt.savefig("samp.png")
    exit()
    """
    """
    prob = sampler.get_log_prob()

    plt.clf()
    for j in range(prob.shape[1]):
        plt.plot(prob[:, j])
    plt.xlabel("iter", fontsize=15)
    plt.ylabel("prob", fontsize=15)
    plt.savefig("prob.png")
    np.savetxt("prob.dat", prob)
    exit()
    """
    return samples, prob

def mcmc(perturb_percent, perturb_this, level):
    def extract_x_results(samp, prob):
        # Shape of samp: ((niter-burn_in)*nwalkers, n x values) 
        # Shape of prob: ((niter-burn_in)*nwalkers)
        vals = np.empty(samp.shape[1])
        where_best = np.argmax(prob)
        
        for k in range(samp.shape[1]):
            vals[k] = samp[where_best, k]
        
        return vals

    print("mcmc", perturb_percent)
    NANT = 4
    NBASELINE = 6
    NUM_TRIALS = 1000

    if perturb_this not in [ "gain", "vis" ]:
        raise ValueError("\"perturb_this\" has unknown value")

    if level == "exact":
        log_func = exact_loglike
    elif level == "approx":
        log_func = approx_loglike
    else:
        raise ValueError("Invalid level in mcmc")

    if perturb_percent is not None:
        val_in = np.empty((NUM_TRIALS, NBASELINE))
        val_out = np.empty((NUM_TRIALS, NBASELINE))
    else:
        val_in = np.empty((NUM_TRIALS, NANT))
        val_out = np.empty((NUM_TRIALS, NANT))

    variance = approximate_variance(perturb_percent, perturb_this, level)
    
    for i in range(NUM_TRIALS):

        print(i, end=" "); sys.stdout.flush()

        vis_values = VisSim(NANT, level=level)
        if i == 0: var_mc = sp.Matrix([variance for i in range(len(vis_values.V))])

        if perturb_percent is not None:
            val_in[i] = vis_values.calculate_bl_gains()     # save
            if perturb_this == "vis":
                p_vis_values = perturb_vis(vis_values, perturb_percent)
            else:
                p_vis_values = perturb_gains(vis_values, perturb_percent)
            samp_exact, prob = run_mcmc(p_vis_values, var_mc, loglike=log_func, iters=2000)

            if i == 0:  # Recalculate variance
                variance = approximate_variance1(samp_exact, p_vis_values, log_func)
                var_mc = sp.Matrix([variance for i in range(len(vis_values.V))])

            p_vis_values.x = extract_x_results(samp_exact, prob)
            val_out[i] = p_vis_values.calculate_bl_gains()
            if level == "exact":
                plot_x_hist(val_in[:i+1], val_out[:i+1],
                    "g_hist_mcmc_"+str(perturb_percent)+".png", "g_in", "g_out")
            else:
                plot_x_hist(val_in[:i+1], val_out[:i+1],
                    "g_approx_hist_mcmc_"+str(perturb_percent)+".png", "g_in", "g_out")
        else:
            val_in[i] = vis_values.x
            samp_exact = run_mcmc(vis_values, var_mc, loglike=log_func, iters=1000)
            val_out[i] = extract_x_results(samp_exact, prob)
            if level == "exact":
                plot_x_hist(val_in[:i+1], val_out[:i+1], "x_hist_mcmc.png")
            else:
                plot_x_hist(val_in[:i+1], val_out[:i+1], "x_approx_hist_mcmc.png")

    print()

