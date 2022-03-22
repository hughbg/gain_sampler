import numpy as np
import pyfftw
from calcs import split_re_im, unsplit_re_im, remove_x_im, restore_x_im

SQRT_2 = np.sqrt(2)

class FFTWPlan:
    def __init__(self, shape, axes, direction):
        assert isinstance(axes, tuple)
        assert isinstance(shape, tuple)
                  
        self.shape = shape
        self.a = pyfftw.empty_aligned(shape, dtype=np.complex128)
        self.b = pyfftw.empty_aligned(shape, dtype=np.complex128)

        self.plan = pyfftw.FFTW(self.a, self.b, axes=axes, direction=direction)  

        
    def execute(self, a):
        assert a.shape == self.a.shape 
        
        if a.dtype == np.complex128 or a.dtype == np.complex64:
            np.copyto(self.a, a)
        elif a.dtype == np.float64:
            np.copyto(self.a, a+np.zeros_like(a)*1j)
        else:
            raise ValueError("Invalid type for array: "+str(a.dtype))
            
        self.plan()
        return self.b
    
    def random(self, dtype):
        
        re = np.random.normal(size=self.shape)
        re -= np.mean(re)
        if dtype == np.float64:
            x = re
        elif dtype == np.complex128:
            im = np.random.normal(size=self.shape)
            im -= np.mean(im)
            x = re+im*1j
        else:
            raise ValueError("Invalid type for array: "+str(a.dtype))
            
        np.copyto(self.a, x)
        self.plan()
        return self.b

class FourierOps:
    
    def __init__(self, ntime, nfreq, nant):
        self.ntime = ntime
        self.nfreq = nfreq
        self.nant = nant
        assert ntime == nfreq
        self.plan_forward = FFTWPlan((nant, ntime, nfreq), (1, 2), direction="FFTW_FORWARD")
        self.plan_backward = FFTWPlan((nant, ntime, nfreq), (1, 2), direction="FFTW_BACKWARD")
            
    def fft2_normed(self, data):
        # data of shape (nant, ntime, nfreq)
        return self.plan_forward.execute(data)/data.shape[1]
        return np.fft.fft2(data)/data.shape[1]   # Normalization to preserve standard deviation not Parsevals
    
    def ifft2_normed(self, data):
        # data of shape (nant, ntime, nfreq)
        return self.plan_backward.execute(data)*data.shape[1]
        return np.fft.ifft2(data)*data.shape[1]  # Normalization to preserve standard deviation not Parsevals
 
    def condense_real_fft(self, a):
        # a: complex 2-D FFT of real values

        assert a.shape[0] == a.shape[1]

        N = a.shape[0]
        assert N%2 == 0

        # Extract the uniq complex values, picking them out of a
        unique = a[0, :N//2+1]     # From first row
        for i in range(1, N//2):                                              # from subsequent rows up to Nyquist
            unique = np.append(unique, a[i])

        unique = np.append(unique, a[N//2, :N//2+1])                          # First few in Nyqust row

        # Split into real/imag and get rid of 4 imaginary zeros
        where = [ 1, N+1, N**2-N+3, N**2+3 ]
        unique = np.delete(split_re_im(unique), where)

        assert unique.size == N**2

        return unique
    
    def F_v(self, v):
        # v is a vector of x values split into re/im and flattened by time/freq
        
        result = np.reshape(v, (self.ntime, self.nfreq, self.nant*2-1))    # Reshape so can add missing x
        result = restore_x_im(result)              # Add the missing x value for each time/freq. (Is missing for DOF fix). It is 0.
        result = unsplit_re_im(result)             # 2 lines reform to complex N-D arrayfor a Fourier transform 
        result = np.moveaxis(result, 2, 0)         # Make the axes (self.nant, self.ntime, self.nfreq) i.e. ant first. The array for the last ant has imag = 0
        result = self.fft2_normed(result)                  
        condensed = self.condense_real_fft(result[-1])       # The last antenna FFT which is of real values needs to be stripped to unique values
        result = np.append(split_re_im(np.ravel(result[:-1])), condensed)                # flatten and split, now length is (self.nant-1)*self.ntime*self.nfreq*2+1*self.ntime*self.nfreq
        assert result.size == (self.nant-1)*self.ntime*self.nfreq*2+self.ntime*self.nfreq

        return result                           # This is a vector of FFT(x) with no values missing
  
    def F_inv_fft(self, fft):
        
        expanded = self.expand_real_fft(fft[-self.ntime*self.nfreq:])     # Generate the full FFT of the last antenna

        result = unsplit_re_im(fft[:-self.ntime*self.nfreq])                       # 3 lines reshape to complex array (self.nant-1, self.ntime, self.nfreq)
        result = np.reshape(result, (self.nant-1, self.ntime, self.nfreq))    
        result = np.append(result, [expanded], axis=0)                     # Tack on last antenna
        result = self.ifft2_normed(result)    
        assert np.sum(result[3].imag) < 1e-15           # All the imag should be 0
        result = np.moveaxis(result, 0, 2)           # Make the axes (self.ntime, self.nfreq, self.nant)
        result = remove_x_im(split_re_im(result))    # Remove one imaginary x value in each time/freq   
        result = np.ravel(result)                    # Flatten, now length is self.ntime*self.nfreq*(self.nant*2-1)
        return result

    
    # Why we can't just put zeros in S for the real FFT components. Because the FFT
    # components are more than just zeros, they have to be conjugates of each other and
    # ordered. This can't be done with the prior.

    def expand_real_fft(self, components):
        N = int(np.sqrt(components.size))
        assert N**2 == components.size, "components is not square: "+str(components.shape)

        # Insert 4 zero values 
        where = [ 1, N, components.size-N+1, components.size]
        unique = np.insert(components, where, [0, 0, 0, 0])

        # Make complex
        unique = unsplit_re_im(unique)

        # Assemble
        fft = np.zeros((N, N), dtype=np.complex128)

        # First row
        fft[0, :N//2+1] = unique[:N//2+1]
        fft[0, N//2+1:N] = np.conj(np.flip(fft[0, 1:N//2]))

        # Subsequent rows half way down and their pair
        start = N//2+1
        for i in range(1, N//2):
            row_start = start+N*(i-1)
            fft[i] = unique[row_start: row_start+N]
        fft[N//2+1:N, 1:N] = np.conj(np.flip(fft[1:N//2, 1:N]))


        # Nyquist row
        fft[N//2, :N//2+1] = unique[-N//2-1:]
        fft[N//2, N//2+1:N] = np.conj(np.flip(fft[N//2, 1:N//2]))

        # The last elements in the first column beyond N//2
        fft[-N//2+1:, 0] = np.conj(np.flip(fft[1:N//2, 0]))

        return fft        
    