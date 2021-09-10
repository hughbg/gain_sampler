import matplotlib.pyplot as plt
import numpy as np

class PerturbedLine:
    def __init__(self, npoints):
        # Numbers to form the likelihood 
        # (s - (ax+b))^T N^-1 (s - (ax+b))
        self.x = np.arange(npoints)+2
        self.b = (np.random.random(1)-0.5)*10
        self.a = (np.random.random(1)-0.5)*5
        self.N_inv = np.diag(np.random.random(npoints))
        
        self.s = self.a*self.x+self.b+(np.random.random(npoints)-0.5)*2
        
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
    

def vec_likelihood_to_scalar(t, a, M):
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
    a_vec = np.full(s.size, a).T
    M = np.dot(np.dot(X.T, N_inv), X)
    
    C1, C2, C3 = vec_likelihood_to_scalar(t, a, M)
    
    mu, sigma_sqr = form_gaussian(C1, C2, C3)
    
    check(mu, sigma_sqr, a, C3, np.dot(np.dot((s-(a*x+b)).T, N_inv), (s-(a*x+b))))
    
    return mu, sigma_sqr    
  
def conditional_b(s, a, x, b, N_inv):
    q = s-a*x
    
    C1, C2, C3 = vec_likelihood_to_scalar(q, b, N_inv)
    
    mu, sigma_sqr = form_gaussian(C1, C2, C3)
    
    check(mu, sigma_sqr, b, C3, np.dot(np.dot((s-(a*x+b)).T, N_inv), (s-(a*x+b))))
    
    return mu, sigma_sqr

def combine_two_gaussians(mu_1, sigma_1_sqr, mu_2, sigma_2_sqr):
    mu = (mu_1*sigma_2_sqr+mu_2*sigma_1_sqr)/(sigma_1_sqr+sigma_2_sqr)
    sigma_sqr = (sigma_1_sqr*sigma_2_sqr)/(sigma_1_sqr+sigma_2_sqr)
    
    return mu, sigma_sqr

p = PerturbedLine(10)

# Priors
mu_a_prior = p.a
sigma_sqr_a_prior = 0.1
mu_b_prior = p.b
sigma_sqr_b_prior = 0.1
               
mu_a, sigma_a_sqr = conditional_a(p.s, p.a, p.x, p.b, p.N_inv)
print(mu_a)
mu_b, sigma_b_sqr = conditional_b(p.s, p.a, p.x, p.b, p.N_inv)

mu, sigma_sqr = combine_two_gaussians(mu_a, sigma_a_sqr, mu_a_prior, sigma_sqr_a_prior)
plot_gaussian(mu, sigma_sqr)
print(p.a, p.b)





