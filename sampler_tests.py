
import numpy as np
from vis_creator import VisSim, VisCal, perturb_vis, select_redundant_group, VisTrue
import corner
import matplotlib.pyplot as plt
from calcs import split_re_im, unsplit_re_im
from gls import gls_solve, reduce_dof, generate_proj, restore_x, generate_proj1
import copy

        
def sample_stats(what, samp):
    for i in range(samp.shape[1]):
        print(what+"_"+str(i)+":", np.mean(samp[:,i]), np.var(samp[:,i]))
        
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 15

def new_x_distribution(v):
    # The model has been updated so get a new distribution.
    # If S is set to None and the model is never changed then
    # the mean of the x distribution will be the GLS solution.

    A = reduce_dof(generate_proj(v.g_bar, v.V_model))  # depends on model
    A = reduce_dof(generate_proj1(v.nvis, v.nant))
    N = np.diag(split_re_im(v.obs_variance))
    d = split_re_im(v.get_reduced_observed())                       # depends on model
    d = split_re_im(v.get_reduced_observed1())
    S = np.eye(A.shape[1])*0.01

    if S is None:
        inv_S = np.zeros((A.shape[1], A.shape[1]))
    else:
        inv_S = np.linalg.inv(S)

    term1 = np.dot(A.T, np.dot(np.linalg.inv(N), A))
    term2 = np.dot(A.T, np.dot(np.linalg.inv(N), d))
    dist_mean = np.dot(np.linalg.inv(inv_S+term1), term2)
    dist_covariance = np.linalg.inv(inv_S+term1)

    return dist_mean, dist_covariance

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

    N = np.diag(split_re_im(v.obs_variance))
    C = np.diag(np.full(split_re_im(v.obs_variance).size, 1))
    d = split_re_im(v.V_obs)
    #V = split_re_im(v.get_calibrated_visibilities())
    V = split_re_im(v.initial_vals.V_model)

    if C is None:
        C = inv_C = np.zeros((v.nvis*2, v.nvis*2))
    else:
        inv_C = np.linalg.inv(C)

    # Only want x values to go +/-10% 
    # Fiddle with the prior widths
    # Plot the distributions mathematically
    # Focus day in the office working on this
    # Equation 17
    # Add noise

    term1 = np.dot(A.T, np.dot(np.linalg.inv(N), A))
    dist_covariance = np.linalg.inv(term1+inv_C)
    term2 = np.dot(A.T, np.dot(np.linalg.inv(N), d))+np.dot(inv_C, V)
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

def probability_of(mean, variance, value):
    value = np.zeros_like(value)
    det_variance = np.prod(np.diag(variance))    # Assume diagonal matrix
    inv_variance = np.linalg.inv(variance)

    return np.e**((-0.5*np.dot(np.dot(value-mean, inv_variance), value-mean)))/np.sqrt((2*np.pi)**mean.size*det_variance)

def bests(x, model, v=None, method="mean"):
    def peak(a):
        # Get the peak 
        hist, bin_edges = np.histogram(a, bins=len(a)//10)
        bins = (bin_edges[1:]+bin_edges[:-1])/2
        return bins[np.argmax(hist)]


    if method == "mean":
        best_x = np.array([ np.mean(x[:, i]) for i in range(x.shape[1]) ])
        best_model = np.array([ np.mean(model[:, i]) for i in range(model.shape[1]) ])    
    
    elif method == "hist":
        best_x = np.array([ peak(x[:, i]) for i in range(x.shape[1]) ])
        best_model = np.array([ peak(model[:, i]) for i in range(model.shape[1]) ])   
    
    elif method == "ml":
        assert x.shape[0] == model.shape[0]
        assert v is not None
        
        vv = copy.deepcopy(v)
        vv.g_bar = vv.initial_vals.g_bar
        vv.V_model = vv.initial_vals.g_bar
        
        best = 1e39
        where_best = 0
        for i in range(x.shape[0]):
            vv.x = unsplit_re_im(restore_x(x[i]))
            vv.V_model = unsplit_re_im(model[i]) 
            chi2 = vv.get_chi2()
            if chi2 < best:
                where_best = i
                best = chi2
            
            best_x = x[where_best]
            best_model = model[where_best]
            
    else:
        raise ValueError("Invalid method")
           
    return best_x, best_model
    

def compare_lists(l1, l2):
    assert len(l1) == len(l2), "compare has bad lists "+str(len(l1))+" "+str(len(l2))
    
    for i in range(len(l1)):
        print(l1[i], l2[i])
        
    print("RMS diff", np.sqrt(np.mean((l1-l2)**2)))
 

def plot_hist(a, fname, label, sigma_prior, orig_val, true_val):
    hist, bin_edges = np.histogram(a, bins=len(a)//50)
    bins = (bin_edges[1:]+bin_edges[:-1])/2
    
    sigma = np.std(a-np.mean(a))
    
    plt.clf()

    plt.plot(bins, hist)
    plt.axvline(orig_val, color="r", label="Orig val")
    plt.axvline(true_val, color="g", label="True val")
    plt.title(label+" sigma: "+str(round(sigma,2))+" sigma_prior: "+str(round(sigma_prior,2)))
    plt.legend()
    plt.savefig(fname+".png")


np.random.seed(99)
#v = VisSim(4, redundant=False)
v = VisCal("/scratch2/users/hgarsden/catall/calibration_points/viscatBC_stretch0.02", time=0, freq=0)
v_true = VisTrue("/scratch2/users/hgarsden/catall/calibration_points/viscatBC_stretch0.02", time=0, freq=0)

# Replace the simulated x with GLS solved x which conforms to
# the degrees of freedom. One imaginary value will be 0.
# The sampling will also conform to this.
#v = perturb_vis(v, 5)


#v.x = v.initial_vals.x = gls_solve(v)

print("orig x", v.x)
print("orig model", v.V_model)
print("orig chi2", v.get_quality())


"""

print("Running bluecal")
# Bluecal just recalculates the model which can be perfect if 
# the model is allowed to be different for each antenna.
v_bl = copy.deepcopy(v)
v.x[:] = 0
for i in range(1000):

    #print(i,)

    # Update model
    v_bl.V_model = v_bl.get_calibrated_visibilities()

    
    # These two lines for redundancy  - different methods
    #v_bl.V_model[:] = np.mean(v_bl.V_model)
    #v_bl.V_model[:] = v_bl.V_model[v.get_best_antenna()]

    v_bl.x = gls_solve(v_bl)
   
    #v_bl.fold_x_into_model()
    
#v_bl.x = gls_solve(v_bl)
   
    
print("bluecal x", v_bl.x)
print("bluecal model", v_bl.V_model)
print("bluecal chi2", v_bl.get_chi2())
print(v_bl.x)
"""

print("Running sampling")

num = 10000

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
    #new_model_sample[:] = np.mean(new_model_sample)      # Redundant only

    # Use the sampled model to change the x sampling distribution, and take a sample
    v_x_sampling.V_model = new_model_sample
    x_dist_mean, x_dist_covariance = new_x_distribution(v_x_sampling)
    all_x[i] = np.random.multivariate_normal(x_dist_mean, x_dist_covariance, 1)  
    new_x_sample = unsplit_re_im(restore_x(all_x[i]))
    
orig_x = split_re_im(v.initial_vals.x)
orig_model = split_re_im(v.initial_vals.V_model)
true_x = (split_re_im(v_true.g_bar)/split_re_im(v.initial_vals.g_bar)-1)
true_model = split_re_im(v_true.V_model)
plot_hist(all_x[:, 0], "x_0", "x_0", 0.1, orig_x[0], true_x[0])
exit()
all_data = np.hstack((all_x, all_model))

# Get the peak of the distributions and see if it creates V_obs
best_x, best_model = bests(all_x, all_model, v, "mean")
print("x recovered vs. orig")
compare_lists(restore_x(best_x), split_re_im(v.initial_vals.x))
print("model recovered vs. orig")
compare_lists(best_model, split_re_im(v.initial_vals.V_model))
v.x = unsplit_re_im(restore_x(best_x))
v.V_model = unsplit_re_im(best_model)
#print("orig obs", orig_v.V_obs)
#print("recovered obs", v.get_simulated_visibilities())
#print("orig obs amplitude", np.abs(orig_v.V_obs))
#print("recovered obs amplitude", np.abs(v.get_simulated_visibilities()))
compare_lists(v.initial_vals.V_obs, v.get_simulated_visibilities())
print("recovered rms",v.get_quality())



# Get log posterior evaluate interpret similar to chi2
# Plot compare plot of true vis vs sampled vis      (V_obs)
#print("Sample stats")
#sample_stats("x", all_x)
#sample_stats("model", all_model)

#print("N:", np.repeat(v.obs_variance, 2))
#print("C:", np.repeat(v.obs_variance, 2))
#print("S:", 1e-4)


# Plot traces
plt.clf()
for i in range(10):
    if i//2 == 0: part = "re"
    else: part = "im"
    plt.plot(all_model[:, i], linewidth=0.06, label="V_"+str(i//2)+" ("+part+")")
plt.title("Trace of first 5 model values split into re/im")
plt.savefig("model_trace.png")


# Plot traces
plt.clf()
for i in range(10):
    if i%2 == 0: part = "re"
    else: part = "im"
    plt.plot(all_x[:, i], linewidth=0.06, label="x_"+str(i//2)+" ("+part+")")

plt.title("Trace of first 5 x values split into re/im")
plt.savefig("x_trace.png")

plt.clf()
plt.plot(all_x[:, 3], linewidth=0.06, label="x_3 ("+part+")")
plt.title("Trace of 1 x value")
plt.savefig("x_1_trace.png")
    

"""
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
"""

partial_data = np.concatenate((all_x[:, :10], all_model[:, :10]), axis=1)


# Plot all dist
part = lambda i : "re" if i%2==0 else "im"
plt.clf()
labels = [ r"$"+part(i)+"(x_"+str(i//2)+")$" for i in range(10) ] + [ r"$"+part(i)+"(V_{model}{"+str(i//2)+"})$" for i in range(10) ]
figure = corner.corner(partial_data, labels=labels, show_titles=True, use_math_text=True, labelpad=0.2)

# Overplot GLS solution
# Extract the axes
axes = np.array(figure.axes).reshape((20, 20))

# Loop over the diagonal
orig_x_5 = split_re_im(v.initial_vals.x)[:10]
orig_model_5 = split_re_im(v.initial_vals.V_model)[:10]
true_x_5 = (split_re_im(v_true.g_bar)/split_re_im(v.initial_vals.g_bar)-1)[:10]
true_model_5 = split_re_im(v_true.V_model)[:10]
orig = np.concatenate((orig_x_5, orig_model_5))

for i in range(orig.size):
    ax = axes[i, i]
    ax.axvline(orig[i], color="r", linewidth=0.5)
    if i < 10:
        ax.axvline(true_x_5[i], color="b", linewidth=0.5)
        print("x (samp vs. true)", orig[i], true_x_5[i])
    else:
        ax.axvline(true_model_5[i-10], color="b", linewidth=0.5)
        print("model (samp vs. true)", orig[i], true_model_5[i-10])
    

plt.tight_layout()
plt.savefig("gibbs_all_dist.png")


f = open("gibbs_sampling.html", "w")
f.write("<h2>Trace of 5 model values (split re/im)</h2><p><img src=model_trace.png width=1200>\n<p>")
f.write("<h2>Trace of 5 x values (split re/im)</h2><p><img src=x_trace.png width=1200>\n<p>")
f.write("<h2>Trace of 1 x value</h2><p><img src=x_1_trace.png width=1200>\n<p>")
f.write("<h2>Corner plot using the 5 model values and 5 x values</h2><p><img src=gibbs_all_dist.png width=2400>\n")
exit()

plt.clf()
plt.hist(all_x[:, 0])
plt.savefig("hist")

