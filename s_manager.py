import numpy as np
from fourier_ops import FourierOps
from gls import split_re_im, unsplit_re_im


class SManager:
    def __init__(self, ntime, nfreq, nant):
        self.ntime = ntime
        self.nfreq = nfreq
        self.nant = nant
        self.S = None
        self.fops = FourierOps(ntime, nfreq, nant)
        
    def generate_S(self, func, view=False):
        x = np.arange(-1, 1, 2.0/self.ntime)
        y = np.arange(-1, 1, 2.0/self.nfreq)
        
        assert x.size == self.ntime and y.size == self.nfreq
        
        data = np.zeros((x.size, y.size))
        for i in range(x.size):
            for j in range(y.size):
                data[i][j] = func(x[i], y[j])
          
        
        fft = np.fft.fftshift(data+data*1j)
        fft[0, 0] = 0            # DC

        S = np.tile(np.ravel(split_re_im(fft)), self.nant-1)      
        self.S = np.append(S, self.fops.condense_real_fft(fft)/np.sqrt(2))  # sqrt(2) factor explained in test_fourier.py

        assert self.S.size == (self.nant-1)*(self.ntime*self.nfreq*2)+self.ntime*self.nfreq
        
        if view:
            fops = FourierOps(self.ntime, self.nfreq, self.nant)
            return self.S, data, fops.ifft2_normed(fft)
                
        return self.S
    
    def sigma(self):
        assert self.S is not None, "No S is set (use generate_S)"
        
        # Use the S but generate +/-ve values around 0
        signs = np.random.normal(size=self.ntime*self.nfreq*2)
        signs = np.where(signs>0, 1, -1)
        sigma_data = np.std(self.S[:self.ntime*self.nfreq*2]*signs)
                    
        ifft = unsplit_re_im(self.S[:self.ntime*self.nfreq*2]*signs)          # lines reshape to complex array for 1 ant (self.ntime, self.nfreq)
        ifft = np.reshape(ifft, (self.ntime, self.nfreq))    
        ifft = self.fops.ifft2_normed(ifft)  
        
        return np.std(ifft.real), np.std(ifft.imag), sigma_data               # should be the same because normalized
    
    def sample(self, what="S", exact=False, last_ant=False):
        """
        Sample one antenna 
        """
        assert self.S is not None, "No S is set (use generate_S)"
        assert what in [ "x", "S" ]
        
        
        if last_ant:
            mean = np.zeros(self.ntime*self.nfreq)
            ant_fft = self.S[-self.ntime*self.nfreq:]
        else:
            mean = np.zeros(self.ntime*self.nfreq*2)
            ant_fft = self.S[:self.ntime*self.nfreq*2]
            
        if exact:
            random_signs = np.random.normal(size=ant_fft.size)
            random_signs = np.where(random_signs>0, 1, -1)
            samples = ant_fft*random_signs
        else:
            samples = np.random.multivariate_normal(mean, np.diag(ant_fft))
            
        if what == "x":                        # do an inverse FFT
            if last_ant:
                fops = FourierOps(self.ntime, self.nfreq, self.nant)
                fft = fops.expand_real_fft(samples)
            else:
                fft = np.reshape(unsplit_re_im(samples), (self.ntime, self.nfreq))

        if what == "x":
            return split_re_im(np.ravel(self.fops.ifft2_normed(fft)))                
        else:
            return samples
   
                        
                
if __name__ == "__main__":
    np.set_printoptions(linewidth=140, precision=4)
    import matplotlib.pyplot as plt
    sm = SManager(16, 16, 4)
    
    dc = lambda x, y: 1 if x==0 and y == 0 else 0
    flat = lambda x, y: 1
    gauss = lambda x, y: np.exp(-0.5*(x**2+y**2)/.05)
    S, data, _ = sm.generate_S(flat, view=True)
    
   


    x = sm.sample("x", exact=True)
    x = np.reshape(unsplit_re_im(x), (16, 16))
    print(np.std(x))
    plt.plot(x.real[10])      # time 0
    plt.savefig("x")
    #plt.show()
        