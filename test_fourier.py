import numpy as np
import pyfftw
from fourier_ops import FourierOps, FFTWPlan
from sampler import Sampler
from vis_creator import VisSim, VisSampling
from calcs import split_re_im, unsplit_re_im, flatten_complex_2d

np.random.seed(0)
SQRT_2 = np.sqrt(2)

def percent_diff(x1, x2):
    x1 = x1[x2>np.max(x2)/1000]      # Ignore values that are very small
    x2 = x2[x2>np.max(x2)/1000]
    #return np.max(np.abs((np.ravel(x1)-np.ravel(x2))/np.max(x2)*100))
    return np.max(np.abs((np.ravel(x1)-np.ravel(x2))/np.ravel(x2)*100))

# Test properties of FFT -------------------


fft128 = FFTWPlan((128, 128), (0, 1), "FFTW_FORWARD")
fft256 = FFTWPlan((256, 256), (0, 1), "FFTW_FORWARD")


### TEST things about py FFT

fft1 = fft128.random(dtype=np.complex128)
fft2 = fft256.random(dtype=np.complex128)
  
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
# Do all above again for real input.

fft1 = fft128.random(dtype=np.float64)
fft2 = fft256.random(dtype=np.float64)

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

fft = fft256.random(dtype=np.complex128)/dim
assert abs(np.std(fft.real)-1)*100 < 0.01 and abs(np.std(fft.imag)-1)*100 < 0.1 

ifft = fops.ifft2_normed(np.array([fft]))

assert abs(np.std(ifft.real)-1)*100 < 0.5 and abs(np.std(ifft.imag)-1)*100 < 0.5


# TEST FourierOps

ntime = 32
nfreq = 32
nant = 4

fops = FourierOps(ntime, nfreq, nant)

### TEST
# Check that FFT forward then back reproduces the original vector
random_x = np.random.random(size=ntime*nfreq*(nant*2-1))
x = fops.F_inv_fft(fops.F_v(random_x))

assert percent_diff(random_x, x) < 0.01

### TEST
# From a perfect simulation, test that the x values converted to Fourier components
# work just as well as the x values in real space, to recover the observation.
vis = VisSampling(VisSim(nant, ntime, nfreq), check_0_im=True, remove_last_x_im=False)
A = vis.generate_proj_sparse2()

# The FFT of x has to be laid out right - do it "by hand"
x = np.reshape(vis.x, (ntime, nfreq, nant))
x = np.moveaxis(x, 2, 0)
s = fops.fft2_normed(x)          

# Convert to Fourier space
condensed = fops.condense_real_fft(s[-1])                       # The last antenna FFT which is of real values needs to be stripped to unique values

s = np.append([ flatten_complex_2d(ant) for ant in s[:-1] ], condensed)     # Form s array needed by inverse FFT

with_fft = A.dot(np.append(fops.F_inv_fft(s), np.zeros(ntime*nfreq)))

reduced_d = vis.reduced_observed_flat                          # get reduced data from sim
percent = percent_diff(with_fft, reduced_d)
assert percent < 1, "percent is "+str(percent)

### TEST
# As for previous test but use the x_flat provided, and FFT it with a fourier_op, nothing "by hand"
vis = VisSampling(VisSim(nant, ntime, nfreq), check_0_im=True, remove_last_x_im=True)
A = vis.generate_proj_sparse2()

s = fops.F_v(vis.x_flat)

with_fft = A.dot(fops.F_inv_fft(s))

reduced_d = vis.reduced_observed_flat                          # get reduced data from sim
percent = percent_diff(with_fft, reduced_d)
assert percent < 1, "percent is "+str(percent)


print("Tests passed")
            
