import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import pickle
from fourier_ops import FourierOps
from calcs import split_re_im, unsplit_re_im


np.set_printoptions(linewidth=180)
np.random.seed(8)

def standard_random_draw(size):
    mean = np.zeros(size)
    cov = np.eye(size)

    return np.random.multivariate_normal(mean, cov)

def is_diagonal(a):
    nz = np.count_nonzero(a - np.diag(np.diagonal(a)))
    return nz == 0

with open('draw.dat', 'rb') as handle:
    draw_data = pickle.load(handle)
  
# These are the sizes in draw_data
ntime = nfreq = 16
nant = 4
    
d = draw_data["d"]
A = draw_data["A"]
N = draw_data["N"]
S = draw_data["S"]

assert is_diagonal(S) and is_diagonal(N), "Diagonal matrices are assumed"

fops = FourierOps(ntime, nfreq, nant)

def do_draw(test_S):
    # test_S must be a vector
    
    sqrt_test_S = np.sqrt(test_S)
    N_inv = np.linalg.inv(N)


    sqrt_S_AFT_Ninv_d = np.multiply(sqrt_test_S, fops.AFT_Ninv_v(A, N, d))     
    sqrt_S_AFT_sqrt_Ninv_omega = np.multiply(sqrt_test_S, fops.AFT_Ninv_v(A, N, standard_random_draw(d.size), sqrt_N=True))  
    rhs = sqrt_S_AFT_Ninv_d+standard_random_draw(test_S.size)+sqrt_S_AFT_sqrt_Ninv_omega

    # Work on the LHS

    square_bracket_term = fops.AFT_Ninv_v(A, N, fops.AF_v(A, sqrt_test_S))        # AF.T N_inv (AF sqrt(S))
    square_bracket_term = np.multiply(sqrt_test_S, square_bracket_term)                   # sqrt_S AF.T N_inv AF sqrt(S)
    #square_bracket_term = np.dot(sqrt_test_S, np.dot(A.T, np.dot(N_inv, np.dot(A, sqrt_test_S))))
    square_bracket_term = np.diag(np.full(test_S.size, 1) + square_bracket_term)

    x_proxy, _ = scipy.sparse.linalg.cg(square_bracket_term, rhs)             # Conjugate gradient for stability

    s = np.multiply(sqrt_test_S, x_proxy)

    x = fops.AF_v(np.eye(A.shape[1]), s)       # Just converts s to x

    x = np.reshape(x, (ntime, nfreq, nant*2-1))

    return x

### TEST. Setup S to have only a DC component for each antenna. Then all the re/im values in an antenna should be the same


S = np.zeros(S.shape[0])
for i in range(nant):
    # Set the re/im component of DC
    S[i*512] = 1
    if i < nant-1: S[i*512+1] = 1          # The last antenna must generate real values so skip this


x = do_draw(S)
for i in range(4):
    assert np.unique(x[:, :, i]).size == 1          # All x values the same 

### TEST. Setup S for the first antenna to include only one low mode.
# Look at plots to verify.


S = draw_data["S"]

fft = np.zeros((ntime, nfreq))
fft[ntime//2+1, nfreq//2+1] = 1
fft = np.fft.fftshift(fft)
fft = fft+fft*1j
fft = np.ravel(split_re_im(fft))
for i in range(fft.size):
    S[i, i] = fft[i]


x = do_draw(np.diag(S))


plt.plot(x[10, :, 0])
plt.plot(x[:, 13, 1])
plt.savefig("mode.png")      # Should show low modes

# TEST. Use S_manager and 1 low mode

from s_manager import SManager

sm = SManager(16, 16, 4)

gauss = lambda x, y: np.exp(-0.5*(x**2+y**2)/.01)
S = sm.generate_S(gauss, modes=1)

x = do_draw(S)

plt.clf()
plt.plot(x[10, :, 0])
plt.plot(x[:, 13, 1])
plt.savefig("mode_S.png")      # Should show low modes


print("Tests passed")






