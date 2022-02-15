import numpy as np
from fourier_ops import FourierOps
from sampler import Sampler
from gls import generate_proj
from vis_creator import VisSim
from calcs import split_re_im, unsplit_re_im

np.random.seed(0)
SQRT_2 = np.sqrt(2)

# Test properties of FFT

def random_fft(dim, only_real_data=False):

    re = np.random.normal(size=(dim, dim))
    re -= np.mean(re)
    if only_real_data:
        x = re
    else:
        im = np.random.normal(size=(dim, dim))
        im -= np.mean(im)
        x = re+im*1j
    
    return np.fft.fft2(x)


fft1 = random_fft(128)
fft2 = random_fft(256)
    
# Check DC component is 0 for input with DC = 0
assert np.abs(fft1[0, 0]) < 1e-10 and np.abs(fft2[0, 0]) < 1e-10

# If the input complex data has sigma(re) = 1 and sigma(im) = 1 (they do) then the
# FFT will be the same, as long as the values are normalized by the dimension.
# Thus doubling the dimension will double the sigma of re and im in the FFT.
# For input data with no DC.
sigma1 = np.std(fft1.real/fft1.shape[0])
sigma2 = np.std(fft2.real/fft2.shape[0])

# Check both sigma(re) close to 1
assert abs(sigma1-1)*100 < 1 and abs(sigma2-1)*100 < 1        # 1% diff   

sigma1 = np.std(fft1.imag/fft1.shape[0])
sigma2 = np.std(fft2.imag/fft2.shape[0])

# Check both sigma(im) close to 1
assert abs(sigma1-1)*100 < 1 and abs(sigma2-1)*100 < 1        # 1% diff   

# For real valued data, things are slightly different. There is a factor of sqrt(2) involved.

fft1 = random_fft(128, only_real_data=True)
fft2 = random_fft(256, only_real_data=True)
    
# Check DC component is 0 for input with DC = 0
assert np.abs(fft1[0, 0]) < 1e-10 and np.abs(fft2[0, 0]) < 1e-10

# If the input real data has sigma(re) = 1 then the
# FFT will be the same, as long as the values are normalized by the dimension and
# multiplied by SQRT_2.
# Thus doubling the dimension will double the sigma of re and im in the FFT.
# For input data with no DC.
sigma1 = np.std(SQRT_2*fft1.real/fft1.shape[0])
sigma2 = np.std(SQRT_2*fft2.real/fft2.shape[0])

# Check both sigma(re) close to 1
assert abs(sigma1-1)*100 < 2 and abs(sigma2-1)*100 < 2        # 2% diff   

sigma1 = np.std(SQRT_2*fft1.imag/fft1.shape[0])
sigma2 = np.std(SQRT_2*fft2.imag/fft2.shape[0])

# Check both sigma(im) close to 1
assert abs(sigma1-1)*100 < 2 and abs(sigma2-1)*100 < 2       # 2% diff   

# These above factors scaling the fft means that the ifft has to undo them.

# Check normalisations in FourierOps
dim = 256
fops = FourierOps(dim, dim, 1)

fft = random_fft(dim)/dim
assert abs(np.std(fft.real)-1)*100 < 0.01 and abs(np.std(fft.imag)-1)*100 < 0.1 

ifft = fops.ifft2_normed(fft)
assert abs(np.std(ifft.real)-1)*100 < 0.5 and abs(np.std(ifft.imag)-1)*100 < 0.5

data = ifft.real

fft = fops.rfft2_normed(data)
assert abs(np.std(fft.real)-1)*100 < 0.5 and abs(np.std(fft.imag)-1)*100 < 0.5 

data = fops.rifft2_normed(fft)
assert abs(np.std(data.real)-1)*100 < 0.5 and abs(np.std(data.imag)) < 1e-10


# TEST FourierOps

ntime = 32
nfreq = 32
nant = 4
nvis = 6

fops = FourierOps(ntime, nfreq, nant)

# TEST
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
assert np.max(percent) < 0.1, "percent is "+str(np.max(percent))
            
# TEST
# From a perfect simulation, test that the x values converted to Fourier components
# work just as well as the x values, to recover the data.
vis = VisSim(nant, ntime, nfreq)   
A = generate_proj(vis.g_bar, vis.project_model())

# The FFT of x has to be laid out right
x = np.reshape(vis.x, (ntime, nfreq, nant))
x = np.moveaxis(x, 2, 0)
s = fops.fft2_normed(x)                  
condensed = fops.condense_real_fft(s[-1])                       # The last antenna FFT which is of real values needs to be stripped to unique values
s = np.append(split_re_im(np.ravel(s[:-1])), condensed)         # flatten and split, now length is (self.nant-1)*self.ntime*self.nfreq*2+1*self.ntime*self.nfreq

with_fft = fops.AF_v(A, s)                                     # this produces a reduced data array

reduced_d = split_re_im(np.ravel(vis.get_reduced_observed()))                          # get reduced data from sim
percent = (with_fft-reduced_d)/reduced_d*100
assert np.max(percent) < 0.5, "percent is "+str(np.max(percent))


print("Tests passed")
            
