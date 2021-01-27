import numpy as np
import sys
import sympy as sp
from plot import plot_x_hist
from vis_creator import VisSim, perturb_gains, perturb_vis


def gls_inv_covariance(proj, Ninv):
    """
    Calculate the generalised least-squares inverse covariance term,
    C^-1 = (M^T N^-1 M)^-1, where M is the projection operator and 
    N is the noise covariance matrix.
    
    Parameters
    ----------
    proj : array_like
        Projection operator.
    
    Ninv : array_like
        Inverse noise covariance array.
    
    Returns
    -------
    Cinv : array_like
        Inverse covariance matrix for linearised gain parameters.
    """
    return np.linalg.inv(np.dot(proj.T, np.dot(Ninv, proj)))


def gls_solution(proj, Ninv, data, inv_cov=None):
    """
    Calculate the generalised least-squares solution for the gain fluctuations. 
    The model is V_ij ~ \bar{g}_i \bar{g}_j V_ij^model (1 + x_i + x_j).
    """
    def rms(v):
        assert len(v.shape) == 1
        return np.sqrt(np.sum(v**2))
   
    # Calculate inverse covariance of solution if needed
    if inv_cov is None:
        inv_cov = gls_inv_covariance(proj, Ninv)
    if np.linalg.cond(inv_cov) > 4:
        print("inv_cov condition")
    
    #print("RMS", rms(np.dot(Ninv, data)), rms(np.dot(proj.T, np.dot(Ninv, data))))
    # Calculate GLS solution
    
    xhat = np.dot(inv_cov, np.dot(proj.T, np.dot(Ninv, data)))
    
    # Return solution and inverse covariance
    return xhat, inv_cov


def gls(perturb_percent, perturb_this):

    def one_line(a):
        s = "[ "
        for x in a: s += "{:.4f}".format(x)+" "
        s += "]"
        return s

    NANT = 4
    NBASELINE = 6
    NUM_TRIALS = 1000

    print("gls", perturb_percent)

    if perturb_this not in [ "gain", "vis" ]:
        raise ValueError("\"perturb_this\" has unknown value")

    if perturb_percent is not None:
        val_in = np.empty((NUM_TRIALS, NBASELINE))
        val_out = np.empty((NUM_TRIALS, NBASELINE))
    else:
        val_in = np.empty((NUM_TRIALS, NANT))
        val_out = np.empty((NUM_TRIALS, NANT))

    for i in range(NUM_TRIALS):

        print(i, end=" "); sys.stdout.flush()

        vis_values = VisSim(NANT, level="approx")   # Can't use level="exact" for GLS
        var_mc = sp.Matrix([0.01 for i in range(len(vis_values.V))])

        if perturb_percent is not None:
            val_in[i] = vis_values.calculate_bl_gains()
            if perturb_this == "gain": p_vis_values = perturb_gains(vis_values, perturb_percent)
            else: p_vis_values = perturb_vis(vis_values, perturb_percent)
            p_vis_values.x = gls_solution(p_vis_values.proj, np.eye(p_vis_values.proj.shape[0]), 
                    p_vis_values.calculate_normalized_observed())[0]
            val_out[i] = p_vis_values.calculate_bl_gains()
            """
            if np.max(np.abs(val_out[i])) > 200: 
                #vis_values.print()
                #p_vis_values.print()
                print("Bad", "proj*x", np.dot(p_vis_values.proj, p_vis_values.x), "x", one_line(p_vis_values.x), "norm vis", one_line(p_vis_values.calculate_normalized_observed()), p_vis_values.calculate_observed()); exit()
            else: print("Good", p_vis_values.calculate_normalized_observed())
            """
            plot_x_hist(val_in[:i+1], val_out[:i+1],
                    "g_hist_gls_"+str(perturb_percent)+".png", "g_in", "g_out")
        else:
            val_in[i] = vis_values.x
            val_out[i] = gls_solution(vis_values.proj, np.eye(vis_values.proj.shape[0]), 
                    vis_values.calculate_normalized_observed())[0]
            plot_x_hist(val_in[:i+1], val_out[:i+1], "x_hist_gls.png")

    print()

