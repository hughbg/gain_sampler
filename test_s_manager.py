import numpy as np
from s_manager import SManager
from calcs import unsplit_re_im

def percent_diff(x1, x2):
    return np.max(np.abs((np.ravel(x1)-np.ravel(x2))/np.ravel(x2)*100))


np.random.seed(87654)

dim = 128

sm = SManager(dim, dim, 4)

gauss = lambda x, y: np.exp(-0.5*(x**2+y**2)/.1)
sm.generate_S(gauss, ignore_threshold=0.0)     # Warning - zeroing some modes will give a skewed sigma for S. It won't match x.


# Test that the sigma of S translates to the same sigma of x, for
# real and imag. The correct fft2 norm has to be done, which is in sample()
# The sigma for real has to be the same as the sigma for imag.

S = sm.sample(what="S", exact=True)                 # sigma of S, first antenna now
S = np.reshape(unsplit_re_im(S), (dim, dim))


S_sigma_re = np.std(S.real[S.real!=0])
S_sigma_im = np.std(S.imag[S.imag!=0])

x = sm.sample(what="x", exact=True)                # sigma of x, first antenna
x = np.reshape(unsplit_re_im(x), (dim, dim))

x_sigma_re = np.std(x.real)
x_sigma_im = np.std(x.imag)

#print(S_sigma_re, S_sigma_im, x_sigma_re, x_sigma_im)


# All the sigmas have to be the same

assert percent_diff(S_sigma_im, S_sigma_re) < 0.5 and percent_diff(x_sigma_re, S_sigma_re) < 1.5 and percent_diff(x_sigma_im, S_sigma_re) < 0.5

# Now do the same thing for the last antenna, which must have x.imag=0. S won't be the same as above due to scaling for this,
# but x.real should be the same as above which is the S sigma


x = sm.sample(what="x", exact=True, last_ant=True)
x = np.reshape(unsplit_re_im(x), (dim, dim))

x_sigma_re1 = np.std(x.real)
x_sigma_im1 = np.std(x.imag)

#print(S_sigma_re, S_sigma_im, x_sigma_re, x_sigma_im)

# Remember x.imag must be 0
assert percent_diff(x_sigma_re1, S_sigma_re) < 1 and np.max(np.abs(x_sigma_im1)) < 1e-8

print("Tests passed")