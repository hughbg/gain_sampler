import matplotlib.pyplot as plt
import numpy as np
import corner
import copy

class PerturbedLine:
    def __init__(self, npoints):
        # Numbers to form the likelihood 
        # (s - (ax+b))^T N^-1 (s - (ax+b))
        self.x = np.arange(npoints)+2
        self.b = (np.random.random(1)-0.5)*10
        self.a = (np.random.random(1)-0.5)*5
        self.N_inv = np.diag(np.random.random(npoints)*0.1)
        
        self.s = (1.2*self.a)*self.x+(self.b*1.2)+(np.random.random(npoints)-0.5)
    
    def rms(self):
        return np.sqrt(np.mean((self.s-(self.a*self.x+self.b))**2))
        
    def print(self):
        print("s", self.s)
        print("x", self.x)
        print("b", self.b)
        print("a", self.a)
        print("N_inv", self.N_inv)
        
    def plot(self):
        plt.plot(p.x, p.s)
        plt.savefig("line")
        
def plot_gaussian(mu, sigma_sqr):
    plt.clf()
    x = np.arange(mu-3*np.sqrt(sigma_sqr), mu+3*np.sqrt(sigma_sqr), 6*np.sqrt(sigma_sqr)/100)
    sigma = np.sqrt(sigma_sqr)
    y = np.exp(-0.5*(mu-x)**2/sigma_sqr)
    plt.plot(x, y)
    plt.savefig("gaussian.png")
    
def plot_non_gaussian_by_a(c1, c2, c3, s, x, b, N_inv):
    # For debugging
    
    a = np.arange(c2-3*np.sqrt(1/c1), c2+3*np.sqrt(1/c1), 6*np.sqrt(1/c1)/100)
    
    # Scalar constants
    y = np.exp(-0.5*(c1*(c2-a)**2 + c3))
    plt.plot(a, y)
    
    # Vectors, should be same as scalar constants
    y = np.exp([ -0.5*np.dot(np.dot((s-(aa*x+b)).T, N_inv), (s-(aa*x+b))) for aa in a ])
    plt.plot(a, y)
    
    # Scalars without c3, not the same
    y = np.exp(-0.5*(c1*(c2-a)**2))
    plt.plot(a, y)
    
    plt.savefig("non_gaussian.png")
    

def vec_likelihood_to_scalar(t, M):
    """
    Given a vector liklihood of the form 
        (t-a)^T M (t-a)
    where t is a vector, a is a scalar, and M is a matrix, reduce it to scalar form
    as a function of a. The scalar form is like this:
    
        C1 (C2-a)^2 + C3
        
    The C's are constants.
    
    Returns:
        C1, C2, C3
    """
    
    f = np.dot(M, np.full(M.shape[0], 1))
    g = np.dot(np.full(M.shape[0], 1), M)

    #print(a**2*np.sum(M) - a*np.dot(t, f+g)+np.dot(np.dot(t, M), t))

    k = np.dot(t, f+g)

    #print(np.sum(M)*( np.dot(t, f+g)/(2*np.sum(M))-a)**2 + np.dot(np.dot(t, M), t) - np.dot(t, f+g)**2/(4*np.sum(M)))
    
    return np.sum(M), np.dot(t, f+g)/(2*np.sum(M)), np.dot(np.dot(t, M), t) - np.dot(t, f+g)**2/(4*np.sum(M))

def form_gaussian(c1, c2 ,c3):
    # The c1, c2 constants are interpreted as sigma, mu of a Gaussian
    mu = c2
    sigma_sqr = 1/c1
        
    return mu, sigma_sqr

def check(mu, sigma_sqr, free_param, c3, correct_val):
    # These two things must be the same. The vector version and the scalar version.
    #print(np.isclose( np.dot(np.dot((s-(a*x+b)).T, N_inv), (s-(a*x+b))), (mu-a)/sigma_sqr+C3) )
    assert np.isclose(correct_val, (mu-free_param)**2/sigma_sqr+c3) 

def conditional_a(s, a, x, b, N_inv):
    p = s-b
    X = np.diag(x)
    t = np.dot(np.linalg.inv(X), p)
    M = np.dot(np.dot(X.T, N_inv), X)
    
    C1, C2, C3 = vec_likelihood_to_scalar(t, M)
    
    mu, sigma_sqr = form_gaussian(C1, C2, C3)
    
    check(mu, sigma_sqr, a, C3, np.dot(np.dot((s-(a*x+b)).T, N_inv), (s-(a*x+b))))
    
    return mu, sigma_sqr    
  
def conditional_b(s, a, x, b, N_inv):
    q = s-a*x
    
    C1, C2, C3 = vec_likelihood_to_scalar(q, N_inv)
    
    mu, sigma_sqr = form_gaussian(C1, C2, C3)
    
    check(mu, sigma_sqr, b, C3, np.dot(np.dot((s-(a*x+b)).T, N_inv), (s-(a*x+b))))
    
    return mu, sigma_sqr

def combine_two_gaussians(mu_1, sigma_1_sqr, mu_2, sigma_2_sqr):
    mu = (mu_1*sigma_2_sqr+mu_2*sigma_1_sqr)/(sigma_1_sqr+sigma_2_sqr)
    sigma_sqr = (sigma_1_sqr*sigma_2_sqr)/(sigma_1_sqr+sigma_2_sqr)
    
    return mu, sigma_sqr

def new_a_distribution(s, a, x, b, N_inv):
    # a is only used for checking
    mu_a_cond, sigma_sqr_a_cond = conditional_a(s, a, x, b, N_inv)
    mu_a, sigma_sqr_a = combine_two_gaussians(mu_a_cond, sigma_sqr_a_cond, mu_a_prior, sigma_sqr_a_prior)
    return mu_a, sigma_sqr_a

def new_b_distribution(s, a, x, b, N_inv):
    # b is only used for checking
    mu_b_cond, sigma_sqr_b_cond = conditional_b(s, a, x, b, N_inv)
    mu_b, sigma_sqr_b = combine_two_gaussians(mu_b_cond, sigma_sqr_b_cond, mu_b_prior, sigma_sqr_b_prior)
    return mu_b, sigma_sqr_b

p = PerturbedLine(10)

# Priors
mu_a_prior = p.a
sigma_sqr_a_prior = 1
mu_b_prior = p.b
sigma_sqr_b_prior = 1
               
num = 5000

all_a = np.zeros(num)      
all_b = np.zeros(num)

likely_a, _ = conditional_a(p.s, p.a, p.x, p.b, p.N_inv)
likely_b, _ = conditional_b(p.s, p.a, p.x, p.b, p.N_inv)
pp = copy.copy(p)
print("orig error", p.rms())
pp.a = likely_a
pp.b = likely_b
print("error using likely a and b", pp.rms())
exit()


new_a_sample = p.a         # Initialize

# Take num samples
for i in range(num):
    # Use the sampled a to change the b sampling distribution, and take a sample
    mu, sigma_sqr = new_b_distribution(p.s, new_a_sample, p.x, p.b, p.N_inv)
    all_b[i] = np.random.normal(mu, sigma_sqr)
    new_b_sample = all_b[i]

    # Use the sampled model to change the x sampling distribution, and take a sample
    mu, sigma_sqr = new_a_distribution(p.s, p.a, p.x, new_b_sample, p.N_inv)
    all_a[i] = np.random.normal(mu, sigma_sqr)
    new_a_sample = all_a[i]
    
def best(a):
    # Get the peak 
    hist, bin_edges = np.histogram(a, bins=100)
    bins = (bin_edges[1:]+bin_edges[:-1])/2
    return bins[np.argmax(hist)]

all_data = np.column_stack((all_a, all_b))


# Plot all dist
plt.clf()
labels = [ "a", "b" ]
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

# Get the peak of the distributions
hist, bin_edges = np.histogram(all_a, bins=100)

best_a = best(all_a)
best_b = best(all_b)
print("a recovered vs. orig", best_a, p.a[0])
print("b recovered vs. orig", best_b, p.b[0])
print("orig error", p.rms())
p.a = best_a
p.b = best_b
print("fitted error", p.rms())








