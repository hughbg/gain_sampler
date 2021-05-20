import numpy as np
from scipy.linalg import sqrtm
from vis_creator import VisSim
from gls import generate_proj, reduce_dof, restore_x
import corner
import matplotlib.pyplot as plt
from calcs import split_re_im, unsplit_re_im
from gls import gls_solve

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

v = VisSim(4)
print("V_obs", np.abs(v.V_obs))
v.x = gls_solve(v)
# GLS solution has the missng x, which is also used for sampling
# Therefore it can be used for comparision with the sampling.
# The V_obs values will stay the same because there is no perturbation.
gls_x = np.copy(v.x)
gls_V_obs = np.copy(v.V_obs)

S = None  # np.eye(v.nant*2-1)
A, where = reduce_dof(generate_proj(v.g_bar, v.V_model))
N = np.eye(v.nvis*2)*1e5
d = split_re_im(v.get_reduced_observed())
#print(np.dot(A, split_re_im(v.x)))
#print(v.get_reduced_observed())
s, var = expected_sample_stats(S, N, A, d)
print("Expected sampler solution for s", unsplit_re_im(restore_x(s, where)))


num = 10000
all_x = np.zeros((num, v.nant*2-1))
all_v = np.zeros((num, v.nvis))

dists = np.zeros(num)
for i in range(num):
    all_x[i] = sample(S, N, A, d)
    v.x = restore_x(all_x[i], where)
    all_v[i] = np.abs(v.get_simulated_visibilities())

print("Best x vectors")
hist, bin_edges = np.histogramdd(all_x, bins=10)
ind = np.where(hist == hist.max())
print(hist.max())
for j in range(len(ind[0])):
    x = np.zeros(len(ind))
    for i in range(len(ind)):
        x[i] = (bin_edges[i][ind[i][j]+1]+bin_edges[i][ind[i][j]])/2
    print(x)
    
print("Best vis vectors")
hist, bin_edges = np.histogramdd(all_v, bins=10)
ind = np.where(hist == hist.max())
print(hist.max())
for j in range(len(ind[0])):
    x = np.zeros(len(ind))
    for i in range(len(ind)):
        x[i] = (bin_edges[i][ind[i][j]+1]+bin_edges[i][ind[i][j]])/2
    print(x)
    
sample_stats("x", all_x)
sample_stats("V_obs", all_v); exit()
#plot_v_histograms(all_v, np.abs(gls_V_obs))

figure = corner.corner(all_x, labels=[r"$re(x_0)$", r"$im(x_0)$", r"$re(x_1)$", r"$im(x_1)$", r"$re(x_2)$", r"$im(x_2)$", r"$re(x_3)$", r"$im(x_3)$"], 
                   show_titles=True, use_math_text=True, labelpad=0.2)

# Overplot GLS solution
# Extract the axes
axes = np.array(figure.axes).reshape((v.nant*2-1, v.nant*2-1))

# Loop over the diagonal
gls_x = split_re_im(gls_x)
for i in range(v.nant*2-1):
    ax = axes[i, i]
    ax.axvline(gls_x[i], color="r", linewidth=0.5)
    
plt.tight_layout()
plt.savefig("x_dist.png")

plt.clf()
figure = corner.corner(all_v, labels=[r"$abs(v_0)$", r"$abs(v_1)$", r"$abs(v_2)$", r"$abs(v_3)$", r"$abs(v_4)$", r"$abs(v_5)$"], #r"$re(v_3)$", r"$im(v_3)$", r"$re(v_4)$", r"$im(v_4)$", r"$re(v_5)$", r"$im(v_5)$"],
                   range=[1, 1, 1, 1, 1, 1], show_titles=True, use_math_text=True, labelpad=0.2) 

# Overplot GLS solution
# Extract the axes
axes = np.array(figure.axes).reshape((v.nvis, v.nvis))

# Loop over the diagonal
for i in range(v.nvis):
    ax = axes[i, i]
    ax.axvline(np.abs(gls_V_obs[i]), color="r", linewidth=0.5)
    
    
plt.tight_layout()
plt.savefig("v_dist.png")

# Check visSim values are random Gaussian

all_x = np.zeros((num, v.nant*2))
all_v = np.zeros((num, v.nvis*2))
for i in range(num):
    v = VisSim(4)
    all_x[i] = split_re_im(v.x)
    all_v[i] = split_re_im(v.get_simulated_visibilities())

plt.clf()

figure = corner.corner(all_x, labels=[r"$re(x_0)$", r"$im(x_0)$", r"$re(x_1)$", r"$im(x_1)$", r"$re(x_2)$", r"$im(x_2)$", r"$re(x_3)$", r"$im(x_3)$"],
                   show_titles=True, use_math_text=True, labelpad=0.2) 
plt.tight_layout()
plt.savefig("random_x.png")

plt.clf()

figure = corner.corner(all_v, labels=[r"$re(v_0)$", r"$im(v_0)$", r"$re(v_1)$", r"$im(v_1)$", r"$re(v_2)$", r"$im(v_2)$", r"$re(v_3)$", r"$im(v_3)$", r"$re(v_4)$", r"$im(v_4)$", r"$re(v_5)$", r"$im(v_5)$"],
                   show_titles=True, use_math_text=True, labelpad=0.2) 
plt.tight_layout()
plt.savefig("random_v.png")


f = open("sampling.html", "w")
f.write("<h2>Sampling</h2>Original x values: "+form(v.x)+"<p>\n")
f.write("Sampled x<p><img src=x_dist.png width=1200>\n")
f.write("<p>Original v values: "+form(v.V_obs)+"<p>\n")
f.write("Sampled v<p><img src=v_dist.png width=1200>\n")
# Show that the reconstructed vis values don't cluster around the 
# Original v values
#for i in range(len(gls_V_obs)):
#    f.write("<img src=v"+str(i)+"_hist.png width=800>\n")
f.write("<h3>Check original x, v, are random</h3>")
f.write("<h3>Random x</h3>\n")
f.write("<img src=random_x.png width=1200>\n")
f.write("<h3>Random V_obs</h3>\n")
f.write("<img src=random_v.png width=1200>\n")
f.close()

        
    


    
    
