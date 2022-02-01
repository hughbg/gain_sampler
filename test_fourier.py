import numpy as np
from fourier_ops import FourierOps
from sampler import Sampler
from gls import generate_proj
from vis_creator import VisSim
from calcs import split_re_im, unsplit_re_im



ntime = 12
nfreq = 12
nant = 4
nvis = 6


fops = FourierOps(ntime, nfreq, nant)


# Prove that AF AF.T N-1 d is equal to A A.T N-1 d. 
# That is, with and without the Fourier transform, go forward and back
# The Fourier transforms cancel out, so not needed.
# Tests that the Fourier transforms cancel out.

A = np.random.normal(size=(ntime*nfreq*nvis*2, ntime*nfreq*(nant*2-1)))
N = np.diag(np.random.normal(size=ntime*nfreq*nvis*2))
d = np.random.normal(size=A.shape[0])

forward = fops.AFT_Ninv_v(A, N, d)
with_fft = np.ravel(split_re_im(fops.AF_v(A, forward)))

without_fft = np.ravel(split_re_im(np.dot(A, np.dot(A.T, np.dot(np.linalg.inv(N), d)))))

# Find the percentage difference but have to get rid of zeros for division
with_fft = with_fft[without_fft!=0]
without_fft = without_fft[without_fft!=0]
percent = (with_fft-without_fft)/without_fft*100
assert np.max(percent) < 0.01, "percent is "+str(np.max(percent))
            
# From a perfect simulation, test that the x values converted to Fourier components
# work just as well as the x values, to recover the data.
vis = VisSim(nant, ntime, nfreq)   
A = generate_proj(vis.g_bar, vis.project_model())

# The FFT of x has to be laid out right
x = np.reshape(vis.x, (ntime, nfreq, nant))
x = np.moveaxis(x, 2, 0)
s = np.fft.fft2(x)                  
condensed = fops.condense_real_fft(s[-1])                       # The last antenna FFT which is of real values needs to be stripped to unique values
s = np.append(split_re_im(np.ravel(s[:-1])), condensed)         # flatten and split, now length is (self.nant-1)*self.ntime*self.nfreq*2+1*self.ntime*self.nfreq

with_fft = fops.AF_v(A, s)                                     # this produces a reduced data array

reduced_d = split_re_im(np.ravel(vis.get_reduced_observed()))                          # get reduced data from sim
percent = (with_fft-reduced_d)/reduced_d*100
assert np.max(percent) < 0.01, "percent is "+str(np.max(percent))


print("Tests passed")
            
