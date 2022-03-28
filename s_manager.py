import numpy as np
from fourier_ops import FourierOps
from calcs import split_re_im, unsplit_re_im


class SManager:
    def __init__(self, ntime, nfreq, nant):
        assert ntime == nfreq
        assert ntime >=4, "Not enough time/freq for effective FFT"
        self.ntime = ntime
        self.nfreq = nfreq
        self.nant = nant
        self.S = None
        self.fops = FourierOps(ntime, nfreq, nant)

    def generate_S(self, func, modes=None, ignore_threshold=0.01, zoom_from=None, scale=1.0, view=False):
        
        def select(a):
            if ignore_threshold == 0 and modes is None: good_locations = np.arange(a.size, dtype=np.int)
            else: good_locations = np.where(a>np.max(a)*ignore_threshold)
            print(good_locations[0].size, "modes selected out of", a.size, "("+str(round(good_locations[0].size/a.size, 2)*100)+"%)")
            vals = np.zeros_like(a)
            vals[good_locations] = a[good_locations]
            return np.array([vals, good_locations])
            
        x = np.arange(-1, 1, 2.0/self.ntime)
        y = np.arange(-1, 1, 2.0/self.nfreq)
        
        if zoom_from is not None:
            x *= x.size/zoom_from[0]
            y *= y.size/zoom_from[1]
        
        assert x.size == self.ntime and y.size == self.nfreq
        
        data = np.zeros((x.size, y.size))
        center_x = data.shape[0]//2
        center_y = data.shape[1]//2
        if modes is None:
            i_start = j_start = 0
            i_end = j_end = x.size
        else:
            i_start = center_x-modes
            j_start = center_y-modes
            i_end = center_x+modes+1
            j_end = center_y+modes+1
            
        for i in range(i_start, i_end, 1):
            for j in range(j_start, j_end, 1):
                data[i][j] = func(x[i], y[j])*scale
                
        assert np.min(data) >= 0, "S cannot contain negative values"
       
        data[center_x, :] = data[:, center_y] = 0      # DC
        fft = np.fft.fftshift(data+data*1j)


        S = np.tile(np.ravel(split_re_im(fft)), self.nant-1)      
        S = np.append(S, self.fops.condense_real_fft(fft)/np.sqrt(2))  # sqrt(2) factor explained in test_fourier.py

        assert S.size == (self.nant-1)*(self.ntime*self.nfreq*2)+self.ntime*self.nfreq
        
        self.S = select(S)        # usable modes, S is now 2-D array, data and usable modes
        
        if view:
            fops = FourierOps(self.ntime, self.nfreq, self.nant)
            return self.S, data, fops.ifft2_normed(fft)
                
        return self.S
    
    def sigma(self):
        assert self.S is not None, "No S is set (use generate_S)"
        
        # Use the S but generate +/-ve values around 0
        signs = np.random.normal(size=self.ntime*self.nfreq*2)
        signs = np.where(signs>0, 1, -1)
        sigma_data = np.std(self.S[0][:self.ntime*self.nfreq*2]*signs)
                    
        ifft = unsplit_re_im(self.S[0][:self.ntime*self.nfreq*2]*signs)          # lines reshape to complex array for 1 ant (self.ntime, self.nfreq)
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
           ant_fft = self.S[0][-self.ntime*self.nfreq:]
        else:
           ant_fft = self.S[0][:self.ntime*self.nfreq*2]
            
        if exact:
            random_signs = np.random.normal(size=ant_fft.size)
            random_signs = np.where(random_signs>0, 1, -1)
            samples = ant_fft*random_signs
        else:
            # Assumes all modes are independent. Theh value is interpreted as a variance
            samples = np.array([ np.random.normal(scale=np.sqrt(var)) if var>0 else 0 for var in ant_fft ])
            
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
    gauss = lambda x, y: np.exp(-0.5*(x**2+y**2)/.01)
    S, data, _ = sm.generate_S(gauss, view=True)
    
    plt.matshow(data)
    plt.savefig("x"); exit()


    x = sm.sample("x", exact=False)
    x = np.reshape(unsplit_re_im(x), (16, 16))
    print(np.std(x))
    plt.plot(x.real[10])      # time 0
    plt.savefig("x")
    #plt.show()
        