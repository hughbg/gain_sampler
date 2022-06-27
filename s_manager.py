import numpy as np
from fourier_ops import FourierOps
from calcs import split_re_im, unsplit_re_im, flatten_complex_2d


class SManager:
    def __init__(self, ntime, nfreq, nant):
        assert ntime == nfreq
        assert ntime >=4, "Not enough time/freq for effective FFT"
        self.ntime = ntime
        self.nfreq = nfreq
        self.nant = nant
        self.S_values = self.usable_modes = None
        self.fops = FourierOps(ntime, nfreq, nant)

    def generate_S(self, func, modes=None, ignore_threshold=0.01, zoom_from=None, scale=1, view=False):
        # ignore_threshold is fractional
        
        def select(a):
            good_locations = np.where(a>np.max(a)*ignore_threshold)[0]
            print(good_locations.size, "modes selected out of", a.size, "("+str(round(good_locations.size/a.size, 2)*100)+"%)", "(zero-valued modes are also ignored)")
            vals = np.zeros_like(a)
            vals[good_locations] = a[good_locations]
            return vals, good_locations
        
        assert 0 <= ignore_threshold and ignore_threshold < 1, "ignore_threshold must be in [0, 1)"
        assert modes is None or (isinstance(modes, int) and modes > 0), "modes must be a positive integer"
        assert (isinstance(scale, float) or isinstance(scale, int)) and scale > 0, "scale must be a positive number"
        assert isinstance(zoom_from, tuple) and (isinstance(zoom_from[0], int) and zoom_from[0] > 0) and \
                (isinstance(zoom_from[1], int) and zoom_from[1] > 0), "zoom_from must be a pair of positive integers"
            
        x = np.arange(-1, 1, 2.0/self.ntime)    # ntime values from [-1, 1)
        y = np.arange(-1, 1, 2.0/self.nfreq)
        
        if zoom_from is not None:
            x *= x.size/zoom_from[0]             # ntime values from [-1, 1)*x.size/zoom_from[0]
            y *= y.size/zoom_from[1]
        
        assert x.size == self.ntime and y.size == self.nfreq
        
        data = np.zeros((x.size, y.size))
        center_x = data.shape[0]//2                    # if ntime=16 this will be 8
        center_y = data.shape[1]//2
        if modes is None:
            i_start = j_start = 0
            i_end = x.size; j_end = y.size
        else:
            i_start = center_x-modes                  
            j_start = center_y-modes
            i_end = center_x+modes+1                  # If ntime=16 and modes=8 this will be [0, 17). The end 17 is too high
            j_end = center_y+modes+1                  # If ntime=16 and modes=4 the range will be [4, 13) giving 4 either side of DC
            
            if i_start < 0 or j_start < 0 or self.ntime < i_end or self.nfreq < j_end:
                print("WARNING: Too many modes requested. All non-zero modes will be used.")
                i_start = j_start = 0
                i_end = x.size; j_end = y.size
                
        for i in range(i_start, i_end, 1):
            for j in range(j_start, j_end, 1):
                data[i][j] = func(x[i], y[j])*scale

            
        assert np.min(data) >= 0, "S cannot contain negative values"
       
        data[center_x, :] = 0
        data[:, center_y] = 0      # DC
        
        assert np.sum(data) > 0, "All modes are zero-valued"
        
        # Now we can control the sigma of the x values that will be generated. See parsevals.ipynb
        # The sigma of the x_real and x_imag values will be the sigma of data. 
        #data *= x_prior_sigma/np.std(data)
        
        fft = np.fft.fftshift(data+data*1j)        
        
        fft_flat = np.concatenate((fft.real, fft.imag), axis=None)    # They'll be flattened
        

        S = np.tile(fft_flat, self.nant-1)      
        S = np.append(S, self.fops.condense_real_fft(fft)/2)  # factor of 2 to make the same sigma 

        assert S.size == (self.nant-1)*(self.ntime*self.nfreq*2)+self.ntime*self.nfreq
        
        self.S_values, self.usable_modes = select(S)        
        
        if view:
            fops = FourierOps(self.ntime, self.nfreq, self.nant)
            return S, data, fops.ifft2_normed(fft)
                
        # The shape of the data in S is: first index is by antenna, for each of these there is an fft of size ntime*nfreq
        # split into re/im and flattened. The split between re/im is a block of the real first, then the imag.
        # However the last fft is condensed because it only ganerates 0 imag values in real space which are ignored. 
        # The last antenna has a mix of fft real and imag values.

    
    def S_full(self):
        assert self.S_values is not None, "S is not set, use generate_S()"
        all_modes = np.zeros_like(self.S_values)
        all_modes[self.usable_modes] = self.S_values[self.usable_modes]
        return all_modes

    
    def sigma(self):
        
        # Use the S but generate +/-ve values around 0
        signs = np.random.normal(size=self.ntime*self.nfreq*2)
        signs = np.where(signs>0, 1, -1)
        sigma_data = np.std(self.S_full()[:self.ntime*self.nfreq*2]*signs)
                    
        ifft = unsplit_re_im(self.S_full[:self.ntime*self.nfreq*2]*signs)          # lines reshape to complex array for 1 ant (self.ntime, self.nfreq)
        ifft = np.reshape(ifft, (self.ntime, self.nfreq))    
        ifft = self.fops.ifft2_normed(ifft)  
        
        return np.std(ifft.real), np.std(ifft.imag), sigma_data               # should be the same because normalized
    
    def sample(self, what="S", exact=False, last_ant=False):
        """
        Sample the FFT modes of one antenna 
        """
        assert self.S_values is not None, "No S is set (use generate_S)"
        assert what in [ "x", "S" ]
       
        
        if last_ant:
            ant_fft = self.S_full()[-self.ntime*self.nfreq:]
        else:
            ant_fft = self.S_full()[:self.ntime*self.nfreq*2]
            
        if exact:
            random_signs = np.random.normal(size=ant_fft.size)
            random_signs = np.where(random_signs>0, 1, -1)
            samples = ant_fft*random_signs
        else:
            # Assumes all modes are independent. The value is interpreted as a variance
            samples = np.array([ np.random.normal(scale=np.sqrt(var)) if var>0 else 0 for var in ant_fft ])
            
        if what == "x":                        # do an inverse FFT
            if last_ant:
                fops = FourierOps(self.ntime, self.nfreq, self.nant)
                fft = fops.expand_real_fft(samples)
            else:
                fft = np.reshape(samples[:self.ntime*self.nfreq]+samples[self.ntime*self.nfreq:]*1j, (self.ntime, self.nfreq))

        if what == "x":
            return flatten_complex_2d(self.fops.ifft2_normed(fft))          
        else:
            return samples
   
                        
                
if __name__ == "__main__":
    np.set_printoptions(linewidth=140, precision=4)
    import matplotlib.pyplot as plt
    sm = SManager(16, 16, 4)
    
    dc = lambda x, y: 1 if x==0 and y == 0 else 0
    flat = lambda x, y: 1
    gauss = lambda x, y: np.exp(-0.5*(x**2+y**2)/.01)
    S, data, _ = sm.generate_S(gauss, modes=8, view=True)

    
    plt.matshow(data)
    plt.savefig("x"); exit()


    x = sm.sample("x", exact=False)
    x = np.reshape(unsplit_re_im(x), (16, 16))
    print(np.std(x))
    plt.plot(x.real[10])      # time 0
    plt.savefig("x")
    #plt.show()
        
