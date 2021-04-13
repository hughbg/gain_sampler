import numpy as np
import scipy.linalg
from calcs import split_re_im, unsplit_re_im

def sample(S, N, A, d):
    """
    Sample a multidimensional Gaussian distribution as described in 
    Eriksen, https://arxiv.org/abs/0709.1058.

    Equation 12, but without multiple frequencies.

    TODO: maybe change these symbols to the GLS symbols.
    """

    if S is None and N is None:
        raise ValueError("S and N cannot both be None")
        
    if S is None:
        S_inv = np.zeros((A.shape[1], A.shape[1]))
    else: 
        S_inv = np.linalg.inv(S)

    if N is None:
        N_inv = np.zeros((A.shape[0], A.shape[0]))
    else: 
        N_inv = np.linalg.inv(N)

    w_0 = np.random.normal(size=S_inv.shape[1])
    w_1 = np.random.normal(size=N_inv.shape[1])
        
    lhs_term = np.linalg.inv(S_inv+np.dot(A.T, np.dot(N_inv, A)))  # inversed
    rhs_term_1 = np.dot(A.T, np.dot(N_inv, d))
    rhs_term_2 = np.dot(scipy.linalg.sqrtm(S_inv), w_0)
    rhs_term_3 = np.dot(A.T, np.dot(scipy.linalg.sqrtm(N_inv), w_1))
    rhs = rhs_term_1+rhs_term_2+rhs_term_3
    s = np.dot(lhs_term,  rhs)
    
    s_var = np.mean(np.diag(lhs_term))
    s_hat = np.mean(np.diag(np.dot(lhs_term, rhs_term_1)))

    return s, s_hat, s_var


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


def gls_solve(vis):
    """
    Calculate the generalised least-squares solution for the gain fluctuations. 
    The model is V_ij ~ \bar{g}_i \bar{g}_j V_ij^model (1 + x_i + x_j).
    
    The x values will be updated in vis.
    
    Parameters
    ----------
    
    vis: VisSim, VisCal, or VisTrue object
    """
    
    def reduce_dof(p):
        best = 1e9
        for i in range(1, p.shape[1], 2):
            cut = np.delete(p, i, axis=1)
            cond = np.linalg.cond(np.dot(cut.T, cut))
            if cond < best:
                cond = best
                where_best = i

        return np.delete(p, where_best, axis=1), where_best

    def restore_x(x_red, where):
        return np.insert(x_red, where, 0)
    
    def generate_proj(g_bar, model):
        
        def separate_real_imag(g0, g1, Vm):
            V = g0*g1*Vm
            Vij_re = (g0*g1*Vm).real
            Vij_im = (g0*g1*Vm).imag
            #print(Vij_re, -Vij_im, Vij_im, Vij_re); exit()
       
            return Vij_re, Vij_im

        # Generate the projection operator
        proj = np.zeros((model.size*2, g_bar.size*2))
        k = 0
        for i in range(g_bar.size):
            for j in range(i+1, g_bar.size):
                re, im = separate_real_imag(g_bar[i], np.conj(g_bar[j]), model[k])

                proj[k*2,i*2] = re; proj[k*2,i*2+1] = -im
                proj[k*2,j*2] = re; proj[k*2,j*2+1] = im

                proj[k*2+1,i*2] = im; proj[k*2+1,i*2+1] = re
                proj[k*2+1,j*2] = im; proj[k*2+1,j*2+1] = -re

                k += 1

        return proj
    
    if vis.level != "approx": 
        raise RuntimeError("GLS can only operate on offsets calculated approximately")
    
    # Convert the variances into an expanded diagonal array 
    Ninv = np.linalg.inv(np.diag(np.repeat(vis.variances, 2)))
    
    # Generate projection matrix
    proj = generate_proj(vis.g_bar, vis.V_model)
    
    # Now remove a column from the proj matrix, which will reduce the
    # number of x values found by 1. This removes a degree of freedom
    # so a solution can be found.
    proj, where_cut = reduce_dof(proj)

    inv_cov = gls_inv_covariance(proj, Ninv)
    
    #print("RMS", rms(np.dot(Ninv, data)), rms(np.dot(proj.T, np.dot(Ninv, data))))
    # Calculate GLS solution
    
    xhat = np.dot(inv_cov, np.dot(proj.T, np.dot(Ninv, split_re_im(vis.get_normalized_observed()))))
    
    # Restore the missing x value and set it to zero. Form complex numbers.
    vis.x = unsplit_re_im(restore_x(xhat, where_cut))      

if __name__ == "__main__":
    from vis_creator import VisSim
    v = VisSim(4)
    gls_solve(v)
    print(v.get_chi2())     # Should be 0 or near 0 because there is no perturbation
