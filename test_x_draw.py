import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import pickle
from fourier_ops import FourierOps
from calcs import split_re_im, unsplit_re_im

np.set_printoptions(linewidth=180)

def standard_random_draw(size):
    mean = np.zeros(size)
    cov = np.eye(size)
    return np.random.multivariate_normal(mean, cov)


with open('draw.dat', 'rb') as handle:
    draw_data = pickle.load(handle)
    
ntime = nfreq = 16
nant = 4
    
d = draw_data["d"]
A = draw_data["A"]
N = draw_data["N"]
S = draw_data["S"]
    
fops = FourierOps(ntime, nfreq, nant)
    
# [1 + sqrt(S)(A.T N-1 A)sqrt(S)]x = sqrt(S) A.TN-1 A d + random + sqrt(S)A.T sqrt(N_inv) random

### TEST. Setup S to have only the same DC component. Then all the x values should be the same
# (except for the one imag value that is zero)

S = np.zeros_like(S)
for i in range(nant):
    S[i*512] = 1
    if i < nant-1: S[i*512+1] = 1          # The last antenna must generate real values so skip this
sqrt_S = S


# Work on the RHS

sqrt_S_AFT_Ninv_d = np.dot(sqrt_S, fops.AFT_Ninv_v(A, N, d))     
sqrt_S_AFT_sqrt_Ninv_omega = np.dot(sqrt_S, fops.AFT_Ninv_v(A, N, standard_random_draw(d.size), sqrt_N=True))  
rhs = sqrt_S_AFT_Ninv_d+standard_random_draw(S.shape[0])+sqrt_S_AFT_sqrt_Ninv_omega

# Work on the LHS

square_bracket_term = fops.AFT_Ninv_v(A, N, fops.AF_v(A, np.sqrt(np.diag(S))))        # AF.T N_inv (AF sqrt(S))
square_bracket_term = np.dot(sqrt_S, square_bracket_term)                   # sqrt_S AF.T N_inv AF sqrt(S)
square_bracket_term = np.diag(np.full(S.shape[0], 1) + square_bracket_term)

x_proxy, _ = scipy.sparse.linalg.cg(square_bracket_term, rhs)             # Conjugate gradient

s = np.dot(sqrt_S, x_proxy)

x = fops.AF_v(np.eye(A.shape[1]), s)       # Just converts s to x

x = np.reshape(x, (ntime, nfreq, nant*2-1))

assert np.unique(x).size == 1              # All x values the same

### TEST. Setup S for the first antenna to include only one low mode.

S = draw_data["S"]

fft = np.zeros((ntime, nfreq))
fft[ntime//2+1, nfreq//2+1] = 1
fft = np.fft.fftshift(fft)
fft = fft+fft*1j
fft = np.ravel(split_re_im(fft))
for i in range(fft.size):
    S[i, i] = fft[i]
sqrt_S = S

# Work on the RHS

sqrt_S_AFT_Ninv_d = np.dot(sqrt_S, fops.AFT_Ninv_v(A, N, d))     
sqrt_S_AFT_sqrt_Ninv_omega = np.dot(sqrt_S, fops.AFT_Ninv_v(A, N, standard_random_draw(d.size), sqrt_N=True))  
rhs = sqrt_S_AFT_Ninv_d+standard_random_draw(S.shape[0])+sqrt_S_AFT_sqrt_Ninv_omega

# Work on the LHS

square_bracket_term = fops.AFT_Ninv_v(A, N, fops.AF_v(A, np.sqrt(np.diag(S))))        # AF.T N_inv (AF sqrt(S))
square_bracket_term = np.dot(sqrt_S, square_bracket_term)                   # sqrt_S AF.T N_inv AF sqrt(S)
square_bracket_term = np.diag(np.full(S.shape[0], 1) + square_bracket_term)

x_proxy, _ = scipy.sparse.linalg.cg(square_bracket_term, rhs)             # Conjugate gradient

s = np.dot(sqrt_S, x_proxy)

x = fops.AF_v(np.eye(A.shape[1]), s)       # Just converts s to x

x = np.reshape(x, (ntime, nfreq, nant*2-1))


plt.plot(x[10, :, 0])
plt.plot(x[:, 13, 1])
plt.savefig("mode.png")      # Should show low modes



