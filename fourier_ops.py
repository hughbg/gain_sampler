import numpy as np
import pyfftw
from calcs import flatten_complex_2d


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
        assert ntime >= 4 and nfreq >= 4, "Not enough times/freqs for FFT"
        self.ntime = ntime
        self.nfreq = nfreq
        self.nant = nant
        assert ntime == nfreq
        self.plan_forward = FFTWPlan((nant, ntime, nfreq), (1, 2), direction="FFTW_FORWARD")
        self.plan_backward = FFTWPlan((nant, ntime, nfreq), (1, 2), direction="FFTW_BACKWARD")
            
    def fft2_normed(self, data):
        # data of shape (nant, ntime, nfreq)
        #return self.plan_forward.execute(data)/data.shape[1]
        return np.fft.fft2(data)/data.shape[1]   # Normalization to preserve standard deviation not Parsevals
    
    def ifft2_normed(self, data):
        # data of shape (nant, ntime, nfreq)
        #return self.plan_backward.execute(data)*data.shape[1]
        return np.fft.ifft2(data)*data.shape[1]  # Normalization to preserve standard deviation not Parsevals
 
    def condense_real_fft(self, a):
        # a: complex 2-D FFT of real values

        assert a.shape[0] == a.shape[1]

        N = a.shape[0]

        assert N%2 == 0

        # Extract the uniq complex values, picking them out of a
        unique = np.empty(N*N//2+2, dtype=np.complex128)

        unique[:N//2+1] = a[0, :N//2+1]     # From first row
        for i in range(1, N//2):                                              # from subsequent rows up to Nyquist
            unique[N//2+1+(i-1)*N : N//2+1+i*N] = a[i]

        unique[-(N//2+1):] = a[N//2, :N//2+1]                          # First few in Nyqust row

        # Split into real/imag and get rid of 4 imaginary zeros
        unique_split = np.empty(unique.size*2)
        unique_split[0::2] = unique.real
        unique_split[1::2] = unique.imag
        where = [ 1, N+1, N**2-N+3, N**2+3 ]
        unique = np.delete(unique_split, where)

        assert unique.size == N**2

        return unique
    
    # Why we can't just put zeros in S for the real FFT components. Because the FFT
    # components are more than just zeros, they have to be conjugates of each other and
    # ordered. This can't be done with the prior.

    def expand_real_fft(self, components):
        N = int(np.sqrt(components.size))
        assert N**2 == components.size, "components is not square: "+str(components.shape)

        # Insert 4 zero values 
        where = [ 1, N, components.size-N+1, components.size]
        unique = np.insert(components, where, [0, 0, 0, 0])

        unique_re = unique[0::2]
        unique_im = unique[1::2]

        # Assemble
        fft_re = np.zeros((N, N))
        fft_im = np.zeros((N, N))

        # First row
        fft_re[0, :N//2+1] = unique_re[:N//2+1]
        fft_im[0, :N//2+1] = unique_im[:N//2+1]
        
        fft_re[0, N//2+1:N] = np.flip(fft_re[0, 1:N//2])
        fft_im[0, N//2+1:N] = -np.flip(fft_im[0, 1:N//2])   # -ve does conjugate

        # Subsequent rows half way down and their pair
        start = N//2+1
        for i in range(1, N//2):
            row_start = start+N*(i-1)
            fft_re[i] = unique_re[row_start: row_start+N]
            fft_im[i] = unique_im[row_start: row_start+N]
        fft_re[N//2+1:N, 1:N] = np.flip(fft_re[1:N//2, 1:N])
        fft_im[N//2+1:N, 1:N] = -np.flip(fft_im[1:N//2, 1:N])


        # Nyquist row
        fft_re[N//2, :N//2+1] = unique_re[-N//2-1:]
        fft_im[N//2, :N//2+1] = unique_im[-N//2-1:]
        
        fft_re[N//2, N//2+1:N] = np.flip(fft_re[N//2, 1:N//2])
        fft_im[N//2, N//2+1:N] = -np.flip(fft_im[N//2, 1:N//2])

        # The last elements in the first column beyond N//2
        fft_re[-N//2+1:, 0] = np.flip(fft_re[1:N//2, 0])
        fft_im[-N//2+1:, 0] = -np.flip(fft_im[1:N//2, 0])

        return np.reshape(fft_re+fft_im*1j, (N, N))
    
    def F_v(self, v):
        # Shape of v: a block of x values re(x_0) size ntime*nfreq, then another block for im(x_0) ...
        
        result = np.reshape(v, (self.nant*2-1, self.ntime, self.nfreq))    # Reshape so can add missing x
        result = np.append(result, [np.zeros((self.ntime, self.nfreq))], axis=0)     # Add the missing x value. It is 0.
        # Now combine the grids into complex numbers
        result = result[0::2]+result[1::2]*1j
        result = self.fft2_normed(result)             
        condensed = self.condense_real_fft(result[-1])       # The last antenna FFT which is of real values needs to be stripped to unique values
        result = np.append([ flatten_complex_2d(f) for f in result[:-1] ], condensed)     # flatten and split
        assert result.size == (self.nant-1)*self.ntime*self.nfreq*2+self.ntime*self.nfreq

        # Shape of result: a block of modes values re(fft(x_0)) size ntime*nfreq, then another block for im(fft(x_0)) ...
        # and for the last antenna it is condensed.
        return result                           
  
    def F_inv_fft(self, fft):
        # Shape of fft is the output of F_v
        expanded = self.expand_real_fft(fft[-self.ntime*self.nfreq:])     # Generate the full FFT of the last antenna
 
        # Lines reshape to complex array (self.nant-1, self.ntime, self.nfreq)
        result = np.reshape(fft[:-self.ntime*self.nfreq], ((self.nant-1)*2, self.ntime, self.nfreq))
        result = result[0::2]+result[1::2]*1j
        result = np.append(result, [expanded], axis=0)                     # Tack on last antenna
        result = self.ifft2_normed(result)    
        assert np.max(np.abs(result[self.nant-1].imag)) < 1e-15           # All the imag should be 0

        # Separate the re/im in each antenna
        sep = np.ravel([ flatten_complex_2d(r) for r in result ])          # Also flattens
        result = sep[:-self.ntime*self.nfreq]                    #  Now length is (self.nant*2-1)*self.ntime*self.nfreq

        return result

    

    
if __name__ == "__main__":
    f = FourierOps(8, 8, 4)
    data = np.random.normal(size=(8, 8))
    np.allclose(f.expand_real_fft1(data).real, f.expand_real_fft(data).real)    
    np.allclose(f.expand_real_fft1(data).imag, f.expand_real_fft(data).imag)