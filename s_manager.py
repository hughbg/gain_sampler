import numpy as np
from fourier_ops import FourierOps
from gls import split_re_im, unsplit_re_im

class SManager:
    def __init__(self, ntime, nfreq, nant):
        self.ntime = ntime
        self.nfreq = nfreq
        self.nant = nant
        self.S = None
        
    def generate_S(self, func, view=False):
        x = np.arange(-1, 1, 2.0/self.ntime)
        y = np.arange(-1, 1, 2.0/self.nfreq)
        
        assert x.size == self.ntime and y.size == self.nfreq
        
        data = np.zeros((x.size, y.size))
        for i in range(x.size):
            for j in range(y.size):
                data[i][j] = func(x[i], y[j])
                
        fft = np.fft.fftshift(data+data*1j)
        fft[0, 0] = 0                           # zero DC
        S = np.tile(np.ravel(split_re_im(fft)), self.nant-1)      
        fops = FourierOps(self.ntime, self.nfreq, self.nant)
        self.S = np.append(S, fops.condense_real_fft(fft)) 

        assert self.S.size == (self.nant-1)*(self.ntime*self.nfreq*2)+self.ntime*self.nfreq
        
        if view:
            return self.S, data, np.fft.ifft2(fft)
                
        return self.S
    
    def sample(self):
        assert self.S is not None, "No S is set (use generate_S)"
        
        mean = np.zeros(self.ntime*self.nfreq*2)
        samples = np.random.multivariate_normal(mean, np.diag(self.S[:self.ntime*self.nfreq*2]))
        fft = np.reshape(unsplit_re_im(samples), (self.ntime, self.nfreq))
        
        
        return np.fft.ifft2(fft)
        
                        
                
if __name__ == "__main__":
    np.set_printoptions(linewidth=140, precision=4)
    import matplotlib.pyplot as plt
    sm = SManager(32, 32, 4)
    
    dc = lambda x, y: 1 if x==0 and y == 0 else 0
    flat = lambda x, y: 1
    gauss = lambda x, y: np.exp(-0.5*(x**2+y**2)/.1)
    S, power, fft_space = sm.generate_S(gauss, view=True)

    
    # to do : dc
    x = sm.sample().real
    plt.plot(x[0])
    plt.savefig("x")
        