import numpy as np
from numpy.linalg import inv
from calcs import split_re_im, unsplit_re_im, BlockMatrix, remove_x_im, restore_x_im

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
    return inv(np.dot(proj.T, np.dot(Ninv, proj)))

def generate_proj(g_bar, model, remove=True):

    def separate_real_imag(g0, g1, Vm):
        V = g0*g1*Vm
        Vij_re = (g0*g1*Vm).real
        Vij_im = (g0*g1*Vm).imag
        #print(Vij_re, -Vij_im, Vij_im, Vij_re); exit()

        return Vij_re, Vij_im

    # Generate the projection operator for each time/freq and merge
    bm = BlockMatrix()
    for time in range(g_bar.shape[0]):
        for freq in range(g_bar.shape[1]):
            proj = np.zeros((model.shape[2]*2, g_bar.shape[2]*2))
            k = 0
            for i in range(g_bar.shape[2]):
                for j in range(i+1, g_bar.shape[2]):
                    re, im = separate_real_imag(g_bar[time, freq, i], np.conj(g_bar[time, freq, j]), model[time, freq, k])

                    proj[k*2,i*2] = re; proj[k*2,i*2+1] = -im
                    proj[k*2,j*2] = re; proj[k*2,j*2+1] = im

                    proj[k*2+1,i*2] = im; proj[k*2+1,i*2+1] = re
                    proj[k*2+1,j*2] = im; proj[k*2+1,j*2+1] = -re

                    k += 1
            
            
            if remove: bm.add(remove_x_im(proj))
            else: bm.add(proj)

    return bm.assemble()

def gls_solve(vis):
    """
    Calculate the generalised least-squares solution for the gain fluctuations. 
    The model is V_ij ~ \bar{g}_i \bar{g}_j V_ij^model (1 + x_i + x_j).
    
    The x values will be updated in vis.
    
    Parameters
    ----------
    
    vis: VisSim, VisCal, or VisTrue object
    """
        
    if vis.level != "approx": 
        raise RuntimeError("GLS can only operate on offsets used approximately")
    
    # Convert the variances into an expanded diagonal array 
    bm = BlockMatrix()
    for time in range(vis.ntime):
        for freq in range(vis.nfreq):
            bm.add((split_re_im(vis.obs_variance[time][freq])))
    N = bm.assemble()
    Ninv = np.linalg.inv(N)
        
    # Generate projection matrix
    proj = generate_proj(vis.g_bar, vis.V_model)

    inv_cov = gls_inv_covariance(proj, Ninv)
    
    #print("RMS", rms(np.dot(Ninv, data)), rms(np.dot(proj.T, np.dot(Ninv, data))))
    # Calculate GLS solution
    
    xhat = np.dot(inv_cov, np.dot(proj.T, np.dot(Ninv, split_re_im(np.ravel(vis.get_reduced_observed())))))
    
    # Restore the missing x value and set it to zero. Form complex numbers and update vis.
    xhat = xhat.reshape(vis.x.shape[0], vis.x.shape[1], -1)
    return unsplit_re_im(restore_x_im(xhat))  


if __name__ == "__main__":
    from vis_creator import VisSim
    vis = VisSim(4, nfreq=2, ntime=1)
    print("Likelihood before gls", vis.get_unnormalized_likelihood(unity_N=True))    
    vis.x = gls_solve(vis)
    print("Likelihood after gls", vis.get_unnormalized_likelihood(unity_N=True))    