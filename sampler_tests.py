import numpy as np
from scipy.linalg import sqrtm, inv as sinv
from vis_creator import VisSim
from gls import generate_proj, reduce_dof, restore_x
import corner
import matplotlib.pyplot as plt
from calcs import split_re_im, unsplit_re_im
from gls import gls_solve
import copy

def inv(x):
    if isinstance(x, (float, np.float)):
        return 1/x
    else:
        return np.linalg.inv(x)

def dot(x, y):
    if isinstance(x, (float, np.float)):
        return x*y
    else:
        return np.dot(x, y)

def sqrt(x):
    if isinstance(x, (float, np.float)):
        return np.sqrt(x)
    else:
        return sqrtm(x)

def tr(x):
    if isinstance(x, (float, np.float)):
        return x
    else:
        return x.T


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
        if isinstance(d, (float, np.float)):
            S_inv = 0
        else:
            S_inv = np.zeros((A.shape[1], A.shape[1]))
    else: 
        if isinstance(d, (float, np.float)):
            S_inv = 1/S
        else:
            S_inv = inv(S)

    if N is None:
        if isinstance(d, (float, np.float)):
            N_inv = 0
        else:
            N_inv = np.zeros((A.shape[0], A.shape[0]))
    else: 
        if isinstance(d, (float, np.float)):
            N_inv = 1/N
        else:
            N_inv = inv(N)
                      
    if isinstance(d, (float, np.float)):
        w_0 = np.random.normal()
        w_1 = np.random.normal()
    else: 
        w_0 = np.random.normal(size=S_inv.shape[1])
        w_1 = np.random.normal(size=N_inv.shape[1])
        
    lhs_term = inv(S_inv+dot(tr(A), dot(N_inv, A)))  # inversed
    rhs_term_1 = dot(tr(A), dot(N_inv, d))
    rhs_term_2 = dot(sqrt(S_inv), w_0)
    rhs_term_3 = dot(tr(A), dot(sqrt(N_inv), w_1))
    rhs = rhs_term_1+rhs_term_2+rhs_term_3
    s = dot(lhs_term,  rhs)

    return s

def expected_sample_stats(S, N, A, d):
    """
    Calculate an approximate scalar mean value (s_hat) and variance
    corresponding to the input parameters.
    """
    
    if S is None:
        if isinstance(d, (float, np.float)):
            S_inv = 0
        else:
            S_inv = np.zeros((A.shape[1], A.shape[1]))
    else: 
        if isinstance(d, (float, np.float)):
            S_inv = 1/S
        else:
            S_inv = inv(S)

    if N is None:
        if isinstance(d, (float, np.float)):
            N_inv = 0
        else:
            N_inv = np.zeros((A.shape[0], A.shape[0]))
    else: 
        if isinstance(d, (float, np.float)):
            N_inv = 1/N
        else:
            N_inv = inv(N)
    
    term1 = dot(tr(A), dot(inv(N), A))
    term2 = inv(S_inv+term1)           
    s_hat = dot(term2, dot(tr(A), dot(N_inv, d)))
    if isinstance(d, (float, np.float)): s_var = term2
    else: s_var = term2
    return s_hat, s_var

def make_noise_variance(Var_s, S, A):
    if S is None:
        if isinstance(Var_s, (float, np.float)):
            S_inv = 0
        else:
            S_inv = np.zeros((A.shape[1], A.shape[1]))
    else: 
        if isinstance(A, (float, np.float)):
            S_inv = 1/S
        else:
            S_inv = inv(S)

    if isinstance(Var_s, (float, np.float)):
        Var_s_inv = 1/Var_s
    else:
        Var_s_inv = inv(Var_s)

    if isinstance(Var_s, (float, np.float)):
        A_T_inv = 1/A
        A_inv = 1/A
    else:
        A_T_inv = inv(A.T)
        A_inv = inv(A)
        
    result = Var_s_inv-S_inv
    result = dot(A_T_inv, result)
    result = dot(result, A_inv)
    return inv(result)



def form(a):
    s = ""
    for c in a:
        s += str(round(c.real, 4))
        s += ", "
        s += str(round(c.imag, 4))
        s += ", "

    return s

def plot_v_histograms(vis, orig_vis):
    for i in range(orig_vis.size):
        plt.clf()
        plt.hist(vis[:, i], bins=50)
        plt.axvline(orig_vis[i], color="red")
        plt.xlabel("Abs(V"+str(i)+")", fontsize=15)
        plt.title("V"+str(i), fontsize=15)
        plt.savefig("v"+str(i)+"_hist.png")
        
def sample_stats(what, samp):
    for i in range(samp.shape[1]):
        print(what+"_"+str(i)+":", np.mean(samp[:,i]), np.var(samp[:,i]))
        
def distance(v1, v2):
    return np.sqrt(np.mean((v1-v2)**2))

plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 15

def sample_x():
    v = VisSim(4, random_seed=99)
    
    # Replace the simulated x with GLS solved x which conforms to
    # the degrees of freedom. One maginary value will be 0.
    # The sampling will also conform to this.
    v.x = gls_solve(v)
       
    # Setup sampling matrices

    S = None  # np.eye(v.nant*2-1)
    A, where = reduce_dof(generate_proj(v.g_bar, v.V_model), True)
    N = np.diag(np.repeat(v.obs_variance, 2))
    d = split_re_im(v.get_reduced_observed())

    num = 10000
    all_x = np.zeros((num, v.nant*2-1))

    # Take num samples 
    for i in range(num):
        all_x[i] = sample(S, N, A, d)
        
    # Provide information about the set of samples
    
    s, var = expected_sample_stats(S, N, A, d)
    print("Expected sampler solution for x", unsplit_re_im(restore_x(s, where)))
    print("Original x", v.x)

    print("Best x vectors")
    hist, bin_edges = np.histogramdd(all_x, bins=10)
    ind = np.where(hist == hist.max())
    for j in range(len(ind[0])):
        x = np.zeros(len(ind))
        for i in range(len(ind)):
            x[i] = (bin_edges[i][ind[i][j]+1]+bin_edges[i][ind[i][j]])/2
        print(x)

    print("Sample stats")
    sample_stats("x", all_x)

    # Make corner plot of samples
    plt.clf()
    labels = [ r"$re(x_"+str(i)+")$" for i in range(all_x.shape[1]) ]
    figure = corner.corner(all_x, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)

    # Overplot GLS solution
    # Extract the axes
    axes = np.array(figure.axes).reshape((v.nant*2-1, v.nant*2-1))

    # Loop over the diagonal
    orig_x = split_re_im(v.x)
    for i in range(v.nant*2-1):
        ax = axes[i, i]
        ax.axvline(orig_x[i], color="r", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("x_dist.png")

def sample_x_1():
    v = VisSim(4, random_seed=99)
    
    # Replace the simulated x with GLS solved x which conforms to
    # the degrees of freedom. One imaginary value will be 0.
    # The sampling will also conform to this.
    v.x = gls_solve(v)
       
    # Setup matrices

    S = None  # np.eye(v.nant*2-1)
    A, where = reduce_dof(generate_proj(v.g_bar, v.V_model), True)
    N = np.diag(np.repeat(v.obs_variance, 2))
    d = split_re_im(v.get_reduced_observed())
    
    if S is None:
        inv_S = np.zeros((A.shape[1], A.shape[1]))
    else:
        inv_S = inv(S)

    term1 = np.dot(A.T, np.dot(inv(N), A))
    term2 = np.dot(A.T, np.dot(inv(N), d))
    dist_mean = np.dot(inv(inv_S+term1), term2)
    dist_covariance = inv(inv_S+term1)

    num = 10000
    all_x = np.random.multivariate_normal(dist_mean, dist_covariance, num)

    # Provide information about the set of samples
    
    s, var = expected_sample_stats(S, N, A, d)
    print("Expected sampler solution for x", dist_mean)
    print("Original x", v.x)
    print(all_x.shape)

    print("Best x vectors")
    hist, bin_edges = np.histogramdd(all_x, bins=10)
    ind = np.where(hist == hist.max())
    for j in range(len(ind[0])):
        x = np.zeros(len(ind))
        for i in range(len(ind)):
            x[i] = (bin_edges[i][ind[i][j]+1]+bin_edges[i][ind[i][j]])/2
        print(x)

    print("Sample stats")
    sample_stats("x", all_x)

    # Make corner plot of samples
    plt.clf()
    labels = [ r"$re(x_"+str(i)+")$" for i in range(all_x.shape[1]) ]
    figure = corner.corner(all_x, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)

    # Overplot GLS solution
    # Extract the axes
    axes = np.array(figure.axes).reshape((v.nant*2-1, v.nant*2-1))

    # Loop over the diagonal
    orig_x = split_re_im(v.x)
    for i in range(v.nant*2-1):
        ax = axes[i, i]
        ax.axvline(orig_x[i], color="r", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("x_dist.png")



def sample_model():
    v = VisSim(4, random_seed=99)
    
    # Setup sampling matrices


    A = np.diag(split_re_im(v.get_baseline_gains()))
    N = np.diag(np.repeat(v.obs_variance, 2))
    C = np.diag(np.repeat(v.obs_variance, 2))
    d = split_re_im(v.V_obs)
    V = split_re_im(v.V_model)
    
    term1 = np.dot(A.T, np.dot(inv(N), A))
    dist_covariance = inv(term1+inv(C))
    term2 = np.dot(A.T, np.dot(inv(N), d))+np.dot(inv(C), V)
    dist_mean = np.dot(dist_covariance, term2)

    num = 10000
    all_v = np.random.multivariate_normal(dist_mean, dist_covariance, num)
         
    # Provide information about the set of samples
    
    print("Expected sampler solution for model", dist_mean)
    print("Original model", v.V_model)

    """
    print("Best model vectors")
    hist, bin_edges = np.histogramdd(all_v, bins=10)
    ind = np.where(hist == hist.max())
    for j in range(len(ind[0])):
        v = np.zeros(len(ind))
        for i in range(len(ind)):
            v[i] = (bin_edges[i][ind[i][j]+1]+bin_edges[i][ind[i][j]])/2
        print(v)
    """

    print("Sample stats")
    sample_stats("model", all_v)

    plt.clf()
    # Make corner plot of samples
    labels = [ r"$re(V_model_{"+str(i)+"})$" for i in range(all_v.shape[1]) ]
    figure = corner.corner(all_v, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)

    # Overplot GLS solution
    # Extract the axes
    axes = np.array(figure.axes).reshape((v.nvis*2, v.nvis*2))

    # Loop over the diagonal
    orig_v = split_re_im(v.V_model)
    for i in range(v.nvis*2):
        ax = axes[i, i]
        ax.axvline(orig_v[i], color="r", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("model_dist.png")
    
    
def gibbs():
    def new_x_distribution(v):
        # The model has been updated so get a new distribution.
        # If S is set to None and the model is never changed then
        # the mean of the x distribution will be the GLS solution.

        A, where = reduce_dof(generate_proj(v.g_bar, v.V_model), True)  # depends on model
        N = np.diag(np.repeat(v.obs_variance, 2))
        d = split_re_im(v.get_reduced_observed())                       # depends on model
        S = np.eye(A.shape[1])*1e-8

        if S is None:
            inv_S = np.zeros((A.shape[1], A.shape[1]))
        else:
            inv_S = inv(S)
   
        term1 = np.dot(A.T, np.dot(inv(N), A))
        term2 = np.dot(A.T, np.dot(inv(N), d))
        dist_mean = np.dot(inv(inv_S+term1), term2)
        dist_covariance = inv(inv_S+term1)

        return dist_mean, dist_covariance, where
    
    def separate_terms(gi, gj, xi, xj):
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
        
    def generate_m_proj(vis):
        proj = np.zeros((vis.nvis*2, vis.nvis*2))
        k = 0
        for i in range(vis.nant):
            for j in range(i+1, vis.nant):
                term1, term2 = separate_terms(vis.g_bar[i], vis.g_bar[j], vis.x[i], vis.x[j])
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
                
            
    def new_model_distribution(v):
        # The x values have been updated so get a new distribution.
        # If the x value has not been changed and C is set to None
        # and N = I then the mean of the distribution will be the 
        # v.V_model
        
        A = generate_m_proj(v)
        N = np.diag(np.repeat(v.obs_variance, 2))
        C = np.diag(np.repeat(v.obs_variance, 2))
        d = split_re_im(v.V_obs)
        V = split_re_im(v.get_calibrated_visibilities())

        if C is None:
            inv_C = np.zeros((v.nvis*2, v.nvis*2))
        else:
            inv_C = inv(C)
    
        term1 = np.dot(A.T, np.dot(inv(N), A))
        dist_covariance = inv(term1+inv_C)
        term2 = np.dot(A.T, np.dot(inv(N), d)+np.dot(inv_C, V))
        dist_mean = np.dot(dist_covariance, term2)
        
        sigma_1 = inv(np.dot(A.T, np.dot(inv(N), A)))
        mu_1 = np.dot(inv(A), d)
        sigma_2 = C
        mu_2 = V
        
        dist_mean = np.dot(np.dot(sigma_2, inv(sigma_1+sigma_2)), mu_1)+np.dot(np.dot(sigma_1, inv(sigma_1+sigma_2)), mu_2)
        dist_covariance = np.dot(sigma_1, np.dot(inv(sigma_1+sigma_2), sigma_2))

        return dist_mean, dist_covariance
    
    
    v = VisSim(4, random_seed=99)
    # Replace the simulated x with GLS solved x which conforms to
    # the degrees of freedom. One imaginary value will be 0.
    # The sampling will also conform to this.
    v.x = gls_solve(v)
    print("orig x", v.x)
    print("orig model", v.V_model)
    orig_v = copy.deepcopy(v)

    num = 40000

    all_x = np.zeros((num, v.nant*2-1))       # -1 because there'll be a missing imaginary value
    all_model = np.zeros((num, v.nvis*2))
    
    v_x_sampling = copy.deepcopy(v)      
    v_model_sampling = copy.deepcopy(v) 
    
    new_x_sample = v_model_sampling.x         # Initialize
    
    # Take num samples
    for i in range(num):
        # Use the sampled x to change the model sampling distribution, and take a sample
        v_model_sampling.x = new_x_sample
        v_dist_mean, v_dist_covariance = new_model_distribution(v_model_sampling)
        all_model[i] = np.random.multivariate_normal(v_dist_mean, v_dist_covariance, 1)
        new_model_sample = unsplit_re_im(all_model[i])
        
        # Use the sampled model to change the x sampling distribution, and take a sample
        v_x_sampling.V_model = new_model_sample
        x_dist_mean, x_dist_covariance, which_im_cut = new_x_distribution(v_x_sampling)
        all_x[i] = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  
        new_x_sample = unsplit_re_im(restore_x(all_x[i], which_im_cut))
           
    all_data = np.hstack((all_x, all_model))
                     
    # Get log poseterior evaluate interpret similar to chi2
    # Plot compare plot of true vis vs sampled vis      (V_obs)
    print("Sample stats")
    sample_stats("x", all_x)
    sample_stats("model", all_model)
    
    print("N:", np.repeat(v.obs_variance, 2))
    print("C:", np.repeat(v.obs_variance, 2))
    print("S:", 1e-4)
    
    
    # Plot traces
    for i in range(all_model.shape[1]):
        plt.clf()
        plt.plot(all_model[:, i], linewidth=0.06)
        plt.savefig("model_trace_"+str(i)+".png")

    # Plot x dist
    plt.clf()
    labels = [ r"$re(x_"+str(i)+")$" for i in range(all_x.shape[1]) ]
    figure = corner.corner(all_x, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)

    # Overplot GLS solution
    # Extract the axes
    axes = np.array(figure.axes).reshape((v.nant*2-1, v.nant*2-1))

    # Loop over the diagonal
    orig_x = split_re_im(orig_v.x)
    for i in range(v.nant*2-1):
        ax = axes[i, i]
        ax.axvline(orig_x[i], color="r", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("gibbs_x_dist.png")
    
    # Plot model dist
    plt.clf()
    # Make corner plot of samples
    labels = [ r"$re(V_{model}{"+str(i)+"})$" for i in range(all_model.shape[1]) ]

    figure = corner.corner(all_model, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)

    # Overplot GLS solution
    # Extract the axes
    axes = np.array(figure.axes).reshape((v.nvis*2, v.nvis*2))

    # Loop over the diagonal
    orig_model = split_re_im(orig_v.V_model)
    for i in range(v.nvis*2):
        ax = axes[i, i]
        ax.axvline(orig_model[i], color="r", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("gibbs_model_dist.png")
    
    
    # Plot all dist
    plt.clf()
    labels = [ r"$re(x_"+str(i)+")$" for i in range(all_x.shape[1]) ] + [ r"$re(V_{model}{"+str(i)+"})$" for i in range(all_model.shape[1]) ]
    figure = corner.corner(all_data, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)

    # Overplot GLS solution
    # Extract the axes
    #axes = np.array(figure.axes).reshape((v.nant*2-1, v.nant*2-1))

    # Loop over the diagonal
    #orig_x = split_re_im(orig_v.x)
    #for i in range(v.nant*2-1):
    #    ax = axes[i, i]
    #    ax.axvline(orig_x[i], color="r", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("gibbs_all_dist.png")

    f = open("gibbs_sampling.html", "w")
    f.write("<h2>Sampling</h2><p>\n")
    f.write("Gibbs sampled x<p><img src=gibbs_x_dist.png width=1200>\n<p>")
    f.write("Gibbs sampled model<p><img src=gibbs_model_dist.png width=1200>\n")
    f.write("Gibbs sampled all<p><img src=gibbs_all_dist.png width=1200>\n<p>")
    for i in range(all_model.shape[1]):
        f.write("Trace<p><img src=model_trace_"+str(i)+".png width=1200>\n")
    f.close()
    
    plt.clf()
    plt.hist(all_x[:, 0])
    plt.savefig("hist")
    
    # Get the peak of the distributions and see if it creates V_obs
    best_x = np.array([ np.mean(all_x[:, i]) for i in range(all_x.shape[1]) ])
    best_model = np.array([ np.mean(all_model[:, i]) for i in range(all_model.shape[1]) ])
    print(best_x, orig_v.x)
    print(best_model, orig_v.V_model)
    v.x = unsplit_re_im(restore_x(best_x, 0))
    v.V_model = unsplit_re_im(best_model)
    print(orig_v.V_obs)
    print(v.get_simulated_visibilities())
    print(np.abs(orig_v.V_obs))
    print(np.abs(v.get_simulated_visibilities()))
    
    
    

def sampler_tests():
    """
    Sample x and the model separately
    """
    sample_x()
    sample_model()
    f = open("sampling.html", "w")
    f.write("<h2>Sampling</h2><p>\n")
    f.write("Sampled x<p><img src=x_dist.png width=1200>\n<p>")
    f.write("Sampled model<p><img src=model_dist.png width=1200>\n<p>")
    f.close()
    
gibbs()
