import numpy as np
import emcee
import copy
from calcs import split_re_im, unsplit_re_im

BURN_IN = 0.5

def estimate_variance(perturb_percent, perturb_this, level):
    """
    Find the variance between the observed visibilities, and visibilities
    obtained from model times gains, after something has been perturbed.
    
    The variance used in the likelihood is important. This function
    simualtes lots of visivilities using the perturbation that is to be
    used for mcmc, gets all the differences (dy) and calculates the variance
    of them. This is then used when the likelihood is calculated. 
    
    Parameters
    ----------
    perturb_percent : float
        Perturb values by this percentage.
        
    perturb_this: str
        What to perturb. Currently "vis" or "gains".
        
    Returns
    -------
    float
        The variance on real and imag values.
    
    """
    nant = 1000
    
    if perturb_percent == 0: return 1e2
 
    # Generate lots of simulated visibilities from simulated
    # model, gains, gain offets
    vis_values = VisSim(nant, level=level)
    
    # Perturb values and calculate the difference between observed
    # visibilities and the originals.
    if perturb_this == "vis":
        vis_values = perturb_vis(vis_values, perturb_percent)
    else:
        vis_values = perturb_gains(vis_values, perturb_percent)

    all_dy_values = vis_values.V_obs-vis_values.get_simulated_visibilities()
    
    return np.mean(np.abs(split_re_im(all_dy_values)**2))    

def loglike(x, proj, V_reduced, var_vec):
    """
    Calculate likelihood.
    
    Parameters
    ----------
    x : array_like
        Gain offset values split re/im.
    vis_values : VisSim, VisCal, or VisTrue object.
        Original visibilties, model, gains. Using the "x" values, 
        calculate new visibilities from model times gains (with "x").
    var_vec : array_like
        Variances for Gaussian distribution of differences.
        
    Returns
    -------
    float
        Likelihood.
    
    """

    # Find the difference in visibilities based on the new x
    dys = V_reduced-np.dot(proj, x)

    # Calculate likelihood
    dchi2 = 0
    for i, dy in enumerate(dys):
        dchi2 += -0.5 * dy**2 * (1./var_vec[i])

    return dchi2

def run_mcmc(proj, V_reduced, variances, iters):
    """
    Parameters
    ----------
    vis_values : VisSim, VisCal, or VisTrue object.
        Contains visibilties, model, gains. 
    var_vec : array_like
        Variances for Gaussian distribution of differences.
    iters : int
        Length of the Markov chain.
        
    Returns
    -------
    array_like, array_like
    
        First array is the MCMC sample chain of shape ((iters-burn_in)*nwalkers, number of "x" values) 
        Second array is the likelihood chain of shape ((iters-burn_in)*nwalkers)
    """

    
    # Set up emcee sampler
    ndim = proj.shape[1]
    nwalkers = ndim * 3
    p0 = 0.01 * np.random.rand(nwalkers, ndim)

    # Set-up sampler object
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, args=[proj, V_reduced, variances])

    # Run sampler
    for sample in sampler.sample(p0, iterations=iters, progress=False):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

    samples = sampler.get_chain(flat=True, discard=int(iters*BURN_IN))
    prob = sampler.get_log_prob(flat=True, discard=int(iters*BURN_IN))

    return samples, prob

